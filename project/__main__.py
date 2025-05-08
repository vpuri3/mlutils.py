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
            n_hidden=cfg.hidden_dim, n_layers=cfg.num_layers,
            n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
            num_slices=cfg.num_slices, act=cfg.act,
        )
    else:
        print(f"No model selected.")
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
            device=device, gnn_loader=False, stats_every=cfg.epochs//10,
            make_optimizer=None, weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
            lossfun=lossfun, clip_grad_norm=cfg.clip_grad_norm, adam_betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        
        # LR scheduler
        if cfg.schedule is None or cfg.schedule == 'ConstantLR':
            kw['lr'] = cfg.learning_rate
        elif cfg.schedule == 'OneCycleLR':
            kw['Schedule'] = 'OneCycleLR'
            kw['lr'] = cfg.learning_rate
            kw['one_cycle_pct_start'] = cfg.one_cycle_pct_start
            kw['one_cycle_div_factor'] = cfg.one_cycle_div_factor
            kw['one_cycle_final_div_factor'] = cfg.one_cycle_final_div_factor
            kw['one_cycle_three_phase'] = cfg.one_cycle_three_phase
        else:
            kw = dict(**kw, Schedule=cfg.schedule, lr=cfg.learning_rate,)

        #-------------#
        # make Trainer
        #-------------#

        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.add_callback('epoch_end', callback)

        #-------------#
        # load snapshot
        #-------------#

        if cfg.restart_file is not None:
            trainer.load(cfg.restart_file)

        #-------------#
        # train
        #-------------#

        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    if cfg.eval:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer = mlutils.Trainer(
            model, _data, data_, make_optimizer=None, device=device
        )
        trainer.make_dataloader()
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
    exp_name: str = 'exp'
    seed: int = 0
    
    # dataset
    dataset: str = 'dummy'
    force_reload: bool = False

    # model
    model_type: int = 0 # -1: MeshGraphNet, 0: Transolver, 1: ClusterAttentionTransformer, 9: SparseTransformer
    act: str = None
    hidden_dim: int = 128
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.
    num_slices: int = 64

    # training arguments
    epochs: int = 100
    batch_size: int = 1
    weight_decay: float = 0e-0
    learning_rate: float = 1e-3
    schedule: str = 'OneCycleLR'
    one_cycle_pct_start:float = 0.10
    one_cycle_div_factor: float = 1e4
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    clip_grad_norm: float = 1.0

if __name__ == "__main__":
    
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    device = mlutils.select_device()

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if not (cfg.train or cfg.eval):
        print("No mode selected. Select one of train, eval")
        exit()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # create a new experiment directory
    if cfg.train and not cfg.eval:
        if cfg.dataset is None:
            print("No dataset selected.")
            exit()

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