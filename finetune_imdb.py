import torch
from torch import nn
import os,sys
from pathlib import Path
from functools import partial
from inspect import isclass
import random
from IPython.core.debugger import set_trace as bk
import pandas as pd
import numpy as np
import datasets
from fastai.text.all import *
from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast, ElectraForPreTraining
from hugdatafast import *
from utils import *

class MyConfig(dict):
  def __getattr__(self, name): return self[name]
  def __setattr__(self, name, value): self[name] = value

def tokenize_sents_max_len(example, cols, max_len, swap=False):
  # Follow BERT and ELECTRA, truncate the examples longer than max length
  tokens_a = hf_tokenizer.tokenize(example[cols[0]])
  tokens_b = hf_tokenizer.tokenize(example[cols[1]]) if len(cols)==2 else []
  _max_length = max_len - 1 - len(cols) # preserved for cls and sep tokens
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= _max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
  if swap:
    tokens_a, tokens_b = tokens_b, tokens_a
  tokens = [hf_tokenizer.cls_token, *tokens_a, hf_tokenizer.sep_token]
  token_type = [0]*len(tokens)
  if tokens_b: 
    tokens += [*tokens_b, hf_tokenizer.sep_token]
    token_type += [1]*(len(tokens_b)+1)
  example['inp_ids'] = hf_tokenizer.convert_tokens_to_ids(tokens)
  example['attn_mask'] = [1] * len(tokens)
  example['token_type_ids'] = token_type
  return example

class SentencePredictor(nn.Module):

  def __init__(self, model, hidden_size, num_class):
    super().__init__()
    self.base_model = model
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(hidden_size, num_class)
    if c.xavier_reinited_outlayer:
      nn.init.xavier_uniform_(self.classifier.weight.data)
      self.classifier.bias.data.zero_()

  def forward(self, input_ids, attention_mask, token_type_ids):
    x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
    return self.classifier(self.dropout(x[:,0,:])).squeeze(-1).float() # if regression task, squeeze to (B), else (B,#class)


def list_parameters(model, submod_name):
  return list(eval(f"model.{submod_name}").parameters())

def hf_electra_param_splitter(model, wsc_trick=False):
  base = 'base_model'
  embed_name = 'embeddings'
  scaler_name = 'embeddings_project'
  layers_name = 'layer'
  output_name = 'classifier'
  
  groups = [ list_parameters(model, f"{base}.{embed_name}") ]
  for i in range(electra_config.num_hidden_layers):
    groups.append( list_parameters(model, f"{base}.encoder.{layers_name}[{i}]") )
  groups.append( list_parameters(model, output_name) )
  if electra_config.hidden_size != electra_config.embedding_size:
    groups[0] += list_parameters(model, f"{base}.{scaler_name}")

  assert len(list(model.parameters())) == sum([ len(g) for g in groups])
  for i, (p1, p2) in enumerate(zip(model.parameters(), [ p for g in groups for p in g])):
    assert torch.equal(p1, p2), f"The {i} th tensor"
  return groups

def get_layer_lrs(lr, decay_rate, num_hidden_layers):
  lrs = [ lr * (decay_rate ** depth) for depth in range(num_hidden_layers+2)]
  if c.original_lr_layer_decays:
    for i in range(1, len(lrs)): lrs[i] *= decay_rate
  return list(reversed(lrs))

class GradientClipping(Callback):
    def __init__(self, clip:float = 0.1):
        self.clip = clip
        assert self.clip
    def after_backward(self):
        if hasattr(self, 'scaler'): self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

def get_imdb_learner(run_name=None, inference=False):

  # Num_epochs
  num_epochs = 3
  
  # Dataloaders
  dls = imdb_dls
  dls.cpu()
  # if isinstance(c.device, str): dls.to(torch.device(c.device))
  # elif isinstance(c.device, list): dls.to(torch.device('cuda', c.device[0]))
  # else: dls.to(torch.device('cuda:0'))

  # Load pretrained model
  discriminator = ElectraForPreTraining.from_pretrained(f"google/electra-{c.size}-discriminator")

  # Seeds & PyTorch benchmark
  torch.backends.cudnn.benchmark = True
  if c.seeds:
    dls[0].rng = random.Random(c.seeds[i]) # for fastai dataloader
    random.seed(c.seeds[i])
    np.random.seed(c.seeds[i])
    torch.manual_seed(c.seeds[i])

  # Create finetuning model
  model = SentencePredictor(discriminator.electra, electra_config.hidden_size, num_class=2)

  # Discriminative learning rates
  splitter = partial( hf_electra_param_splitter, wsc_trick=False)
  layer_lrs = get_layer_lrs(lr=c.lr, 
                            decay_rate=c.layer_lr_decay,
                            num_hidden_layers=electra_config.num_hidden_layers,)
  
  # Optimizer
  if c.adam_bias_correction: opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=c.weight_decay)
  else: opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=c.weight_decay)
  
  # Learner
  learn = Learner(dls, 
                  model,
                  loss_func=CrossEntropyLossFlat(), 
                  opt_func=opt_func,
                  metrics=accuracy,
                  splitter=splitter if not inference else trainable_params,
                  lr=layer_lrs if not inference else defaults.lr,
                  path='./checkpoints/imdb',
                  model_dir=c.group_name,)

  # Multi gpu
  if isinstance(c.device, list) or c.device is None:
    learn.create_opt()
    learn.model = nn.DataParallel(learn.model, device_ids=c.device)

  # Mixed precision
  learn.to_native_fp16(init_scale=2.**14)

  # Gradient clip
  learn.add_cb(GradientClipping(1.0))
  
  # Logging
  # Logging
  # if run_name and not inference:
  #   if c.logger == 'neptune':
  #     neptune.create_experiment(name=run_name, params={'task':'imdb', **c, **hparam_update})
  #     learn.add_cb(LightNeptuneCallback(False))
  #   elif c.logger == 'wandb':
  #     wandb_run = wandb.init(name=run_name, project='electra_glue', config={'task':'imdb', **c, **hparam_update}, reinit=True)
  #     learn.add_cb(LightWandbCallback(wandb_run))

  # Learning rate schedule
  if c.schedule == 'one_cycle': 
    return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs)
  elif c.schedule == 'adjusted_one_cycle':
    return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs, div=1e5, pct_start=0.1)
  else:
    lr_shed_func = linear_warmup_and_then_decay if c.schedule=='separate_linear' else linear_warmup_and_decay
    lr_shedule = ParamScheduler({'lr': partial(lr_shed_func,
                                               lr_max=np.array(layer_lrs),
                                               warmup_pct=0.1,
                                               total_steps=num_epochs*(len(dls.train)))})
    return learn, partial(learn.fit, n_epoch=num_epochs, cbs=[lr_shedule])

c = MyConfig({
  'device': 'cuda:0', #List[int]: use multi gpu (data parallel)
  # run [start,end) runs, every run finetune every GLUE tasks once with different seeds.
  'start':0,
  'end': 10,
  
  'pretrained_checkpoint': None, # None to use pretrained ++ model from HuggingFace
  'seeds': None,

  'weight_decay': 0,
  'adam_bias_correction': False,
  'xavier_reinited_outlayer': True,
  'schedule': 'original_linear',
  'original_lr_layer_decays': True,
  'double_unordered': True,
  
  # whether to do finetune or test
  'do_finetune': True, # True -> do finetune ; False -> do test
  # finetuning checkpoint for testing. These will become "ckp_dir/{task}_{group_name}_{th_run}.pth"
  
  'size': 'small',
  
  'num_workers': 0,
  'my_model': False, # True only for my personal research
  'logger': 'wandb',
  'group_name': None, # the name of represents these runs
  # None: use name of checkpoint.
  # False: don't do online logging and don't save checkpoints
})

""" Vanilla ELECTRA settings
'adam_bias_correction': False,
'xavier_reinited_outlayer': True,
'schedule': 'original_linear',
'original_lr_layer_decays': True,
'double_unordered': True,
"""

# Check
# if not c.do_finetune: assert c.th_run['mnli'] == c.th_run['ax']
if c.pretrained_checkpoint is None: assert not c.my_model
assert c.schedule in ['original_linear', 'separate_linear', 'one_cycle', 'adjusted_one_cycle']

# Settings of different sizes
if c.size == 'small': c.lr = 3e-4; c.layer_lr_decay = 0.8; c.max_length = 128
elif c.size == 'base': c.lr = 1e-4; c.layer_lr_decay = 0.8; c.max_length = 512
elif c.size == 'large': c.lr = 5e-5; c.layer_lr_decay = 0.9; c.max_length = 512
else: raise ValueError(f"Invalid size {c.size}")
if c.pretrained_checkpoint is None: c.max_length = 512 # All public models is ++, which use max_length 512

# huggingface/transformers
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-discriminator")
electra_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')

# Path
Path('./datasets').mkdir(exist_ok=True)
Path('./checkpoints/imdb').mkdir(exist_ok=True, parents=True)
Path('./test_outputs/imdb').mkdir(exist_ok=True, parents=True)
c.pretrained_ckp_path = Path(f'./checkpoints/pretrain/{c.pretrained_checkpoint}')

# Print info
print(f"process id: {os.getpid()}")
print(c)

imdb_dsets = {}; 
imdb_dls = {}

# Load / download datasets.
dsets_train = datasets.load_dataset("imdb")
# dsets_val = datasets.load_dataset("imdb", split="test")

# Load / Make tokenized datasets
tok_func = partial(tokenize_sents_max_len, max_len=c.max_length, cols=['text'])
imdb_dsets = dsets_train.my_map(tok_func, cache_file_names=f"tokenized_{c.max_length}_{{split}}")
# imdb_dsets['test'] = dsets_val.my_map(tok_func)

# Load / Make dataloaders
hf_dsets = HF_Datasets(imdb_dsets, hf_toker=hf_tokenizer, n_inp=3,
              cols={'inp_ids':TensorText, 'attn_mask':noop, 'token_type_ids':noop, 'label':TensorCategory})

imdb_dls = hf_dsets.dataloaders(bs=32, shuffle_train=True, num_workers=c.num_workers,
                                      cache_name=f"dl_{c.max_length}_{{split}}.json",)

if c.do_finetune:
  for i in range(c.start, c.end):
    learn, fit_fc = get_imdb_learner()
    fit_fc()
    # if run_name: learn.save(f"imdb_{i}")
