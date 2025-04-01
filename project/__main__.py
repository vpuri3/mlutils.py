#
import torch

import os
import yaml
from jsonargparse import CLI
from dataclasses import dataclass

# local
import mlutils
import project # this project

#======================================================================#
import socket
MACHINE = socket.gethostname()

if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    raise Warning(f"DATADIR_BASE for {MACHINE} not set. Defaulting to data/")
    DATADIR_BASE = 'data/'

#======================================================================#

PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
def main(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#
    
    if cfg.dataset == 'dummy':
        import numpy as np

        DATADIR = os.path.join(DATADIR_BASE, 'dummy')
        dataset = project.DummyDataset(DATADIR, transform=project.NormalizeTransform())

        ci = 2
        co = 1

    else:
        print(f"Dataset {cfg.dataset} not found.")
        exit()

    _data, data_ = torch.utils.data.random_split(dataset, [0.80, 0.20])
    
    if GLOBAL_RANK == 0:
        print(f"Loaded {cfg.dataset} dataset with {len(dataset)} cases.")
        print(f"Split into {len(_data)} train and {len(data_)} test cases.")
    
    #=================#
    # MODEL
    #=================#

    if cfg.model_type == 0:
        model = project.TS(
            in_dim=ci, out_dim=co,
            n_hidden=cfg.width, n_layers=cfg.num_layers,
            n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
            num_slices=cfg.num_slices,
        )
    else:
        print(f"No conditioned model selected.")
        raise NotImplementedError()

    #=================#
    # TRAIN
    #=================#

    lossfun  = torch.nn.MSELoss()
    callback = project.RelErrorCallback(case_dir,)

    if cfg.train and cfg.epochs > 0:

        _batch_size  = cfg.batch_size
        batch_size_  = len(data_)
        _batch_size_ = len(_data)

        kw = dict(
            device=device, stats_every=cfg.epochs//10,
            Opt='AdamW', weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
            lossfun=lossfun, Schedule=cfg.schedule, lr=cfg.learning_rate,
        )
        
        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.add_callback('epoch_end', callback)

        if cfg.restart_file is not None:
            trainer.load(cfg.restart_file)

        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    if device != 'cpu' and device != torch.device('cpu'):
        torch.cuda.empty_cache()
    trainer = mlutils.Trainer(model, _data, data_, device=device)
    callback.load(trainer)
    callback(trainer, final=True)

    return

#======================================================================#
@dataclass
class Config:
    '''
    Test project
    '''

    # case configuration
    train: bool = False
    eval: bool = False
    restart_file: str = None
    dataset: str = 'dummy'
    exp_name: str = 'exp'
    seed: int = 123

    # model
    model_type: int = 0
    width: int = 128
    num_layers: int = 5
    num_heads: int = 8
    mlp_ratio: float = 2.0
    num_slices: int = 32

    # training arguments
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    schedule: str = None
    
    batch_size: int = 16

if __name__ == "__main__":
    
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    device = mlutils.select_device()

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if not (cfg.train or cfg.eval):
        print("No mode selected. Select one of train, eval")
        exit()

    if cfg.dataset is None:
        print("No dataset selected.")
        exit()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # create a new experiment directory
    if cfg.train:
        if os.path.exists(case_dir):
            nd = len([dir for dir in os.listdir(CASEDIR) if dir.startswith(cfg.exp_name)])
            cfg.exp_name = cfg.exp_name + '_' + str(nd).zfill(2)
            case_dir = os.path.join(CASEDIR, cfg.exp_name)

        if DISTRIBUTED:
            torch.distributed.barrier()

        if GLOBAL_RANK == 0:
            os.makedirs(case_dir)
            config_file = os.path.join(case_dir, 'config.yaml')
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)

    # load config from experiment directory
    if cfg.eval:
        assert os.path.exists(case_dir)
        config_file = os.path.join(case_dir, 'config.yaml')
        _cfg = cfg
        if GLOBAL_RANK == 0:
            print(f'Loading config from {config_file}')
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg = Config(**cfg)
        cfg.eval = True
        cfg.train = False

    if DISTRIBUTED:
        torch.distributed.barrier()

    main(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#
