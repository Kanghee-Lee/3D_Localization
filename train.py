# -*- coding: future_fstrings -*-
import open3d as o3d  # prevent loading error

import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from lib.data_loaders import make_data_loader
from config import get_config

from lib.trainer import ContrastiveLossTrainer, HardestContrastiveLossTrainer, \
    TripletLossTrainer, HardestTripletLossTrainer, ContrastiveLossTrainer_MLP

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def get_trainer(trainer):
  if trainer == 'ContrastiveLossTrainer':
    return ContrastiveLossTrainer
  elif trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'TripletLossTrainer':
    return TripletLossTrainer
  elif trainer == 'HardestTripletLossTrainer':
    return HardestTripletLossTrainer
  elif trainer == 'ContrastiveLossTrainer_MLP':
    return ContrastiveLossTrainer_MLP
  else:
    raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):

  if config.fcgf_extract :
    assert config.batch_size == 1, 'batch size should be 1'
    assert config.model == 'ResUNetBN2C_extract', 'Inappropriate model for fcgf extract'
    train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_threads=config.train_num_thread)

    val_loader = make_data_loader(
      config,
      config.val_phase,
      config.val_batch_size,
      num_threads=config.val_num_thread)

    Trainer = get_trainer(config.trainer)
    trainer = Trainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
      freeze=config.freeze)
    trainer._extract()

  else :
    train_loader = make_data_loader(
        config,
        config.train_phase,
        config.batch_size,
        num_threads=config.train_num_thread)

    if config.test_valid:
      val_loader = make_data_loader(
          config,
          config.val_phase,
          config.val_batch_size,
          num_threads=config.val_num_thread)
    else:
      val_loader = None

    Trainer = get_trainer(config.trainer)
    if config.freeze :
      trainer = Trainer(
          config=config,
          data_loader=train_loader,
          val_data_loader=val_loader,
          freeze=True)
    else :
      trainer = Trainer(
          config=config,
          data_loader=train_loader,
          val_data_loader=val_loader,
          freeze=False)
    if config.gp_reg :
      trainer.train_reg()
    elif config.load_fcgf :
      trainer.train_mlp()
    else :
      trainer.train()
  #
  # if config.freeze :
  #     trainer.train(freeze=True)
  # trainer.train(freeze=False)


if __name__ == "__main__":
  print('1' * 50)
  print(torch.cuda.device_count())
    # 1
  print(torch.cuda.get_device_name(0))
    # GeForce RTX 2080 Ti
  print(torch.cuda.is_available())
  print('1'*50)

  print(torch.__version__)


  logger = logging.getLogger()
  config = get_config()

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint_32_att_8_1.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)
