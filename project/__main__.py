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
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
import socket
MACHINE = socket.gethostname()

if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti 11 GB
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
else:
    DATADIR_BASE = os.path.join(PROJDIR, 'data')

os.environ["HF_HOME"] = os.path.join(DATADIR_BASE, 'huggingface')

#======================================================================#
def main(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#

    _data, data_, metadata = project.load_dataset(cfg.dataset, DATADIR_BASE)
    in_dim = metadata['in_dim']
    out_dim = metadata['out_dim']

    if GLOBAL_RANK == 0:
        print(f"Loaded {cfg.dataset} dataset with {len(_data)} train and {len(data_)} test cases.")

    #=================#
    # MODEL
    #=================#

    if cfg.model_type == 0:

        if GLOBAL_RANK == 0:
            print(
                f"Using Transformer(in_dim={in_dim}, out_dim={out_dim}) with\n"
                + f"\tchannel_dim={cfg.channel_dim}\n"
                + f"\tnum_blocks={cfg.num_blocks}\n"
                + f"\tnum_heads={cfg.num_heads}\n"
                + f"\tmlp_ratio={cfg.mlp_ratio}\n"
                + f"\tact={cfg.act}\n"
            )

        model = project.Transformer(
            in_dim=in_dim,
            out_dim=out_dim,
            channel_dim=cfg.channel_dim,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            act=cfg.act,
        )
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not implemented.")

    #=================#
    # MAKE TRAINER
    #=================#

    #----------#
    # callback
    #----------#

    callback = project.RelL2Callback(case_dir,)

    #----------#
    # batch_size
    #----------#

    _batch_size = cfg.batch_size
    batch_size_ = _batch_size_ = _batch_size // WORLD_SIZE * 2

    #----------#
    # lossfun
    #----------#

    lossfun = torch.nn.MSELoss()
    batch_lossfun = None # lambda trainer, model, batch: lossfun(model(batch[0]), batch[1])
    statsfun = project.RelL2StatsFun()

    #----------#
    # Trainer kwargs
    #----------#

    if cfg.optimizer == 'adamw':
        make_optimizer = project.make_optimizer_adamw
    elif cfg.optimizer == 'lion':
        make_optimizer = project.make_optimizer_lion
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented.")

    kw = dict(
        device=device, mixed_precision=cfg.mixed_precision, attn_backend=cfg.attn_backend, stats_every=cfg.epochs//10,
        # batch size
        _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
        # optimizer
        make_optimizer=make_optimizer, weight_decay=cfg.weight_decay, epochs=cfg.epochs,
        lossfun=lossfun, batch_lossfun=batch_lossfun, clip_grad_norm=cfg.clip_grad_norm,
        opt_betas=(cfg.opt_beta1, cfg.opt_beta2), opt_eps=cfg.opt_eps,
        # dataloader kwargs
        num_workers=cfg.num_workers, prefetch_factor=2,
        # statistics
        statsfun=statsfun,
    )

    #----------#
    # LR scheduler
    #----------#

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

    if cfg.restart:
        callback.load(trainer)

    #=================#
    # TRAIN
    #=================#

    if cfg.train and cfg.epochs > 0:
        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    if cfg.evaluate:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer.make_dataloader()
        callback.load(trainer)
        callback(trainer, final=True)

    return

#======================================================================#
@dataclass
class Config:
    '''
    Template codebase for training and evaluating models.

    For training, run 

        python -m project --train true ... <CONFIG>

    and the result will be saved to out/<exp_name>/ckpt<01, 02, ...>.

    For evaluation, run

        python -m project --evaluate true --exp_name <exp_name>

    and the model in the latest checkpoint out/<exp_name>/ckptXX will be evaluated.

    For restarting from checkpoint, run

        python -m project --restart true --exp_name <exp_name>
        
    and training will resume from the latest checkpoint in out/<exp_name>/ckptXX.
    '''

    # case configuration
    train: bool = False
    evaluate: bool = False
    restart: bool = False

    exp_name: str = 'exp'
    seed: int = 0

    # dataset
    dataset: str = 'dummy'
    num_workers: int = 0

    # training arguments
    epochs: int = 100
    batch_size: int = 4
    weight_decay: float = 0e-0
    learning_rate: float = 1e-3
    schedule: str = 'OneCycleLR'
    one_cycle_pct_start:float = 0.10
    one_cycle_div_factor: float = 1e4
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = False
    opt_beta1: float = 0.9
    opt_beta2: float = 0.999
    opt_eps: float = 1e-8
    clip_grad_norm: float = 1.0
    optimizer: str = 'adamw' # adamw, lion
    mixed_precision: bool = True
    attn_backend: str = None

    # model
    model_type: int = 0 # 0: Transformer

    num_blocks: int = 4
    channel_dim: int = 64
    num_heads: int = 8
    mlp_ratio: float = 4.0
    act: str = None

#======================================================================#
if __name__ == "__main__":

    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    device = mlutils.select_device()

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if (cfg.train + cfg.evaluate + cfg.restart) != 1:
        msg = f"Invalid mode selection. Select one of train (got {cfg.train}), evaluate (got {cfg.evaluate}), restart (got {cfg.restart})."
        raise ValueError(msg)

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    if cfg.train:
        if os.path.exists(case_dir):
            # if exp_name already exists, append a number to make it unique
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
    if cfg.evaluate or cfg.restart:
        assert os.path.exists(case_dir), f"Experiment directory {case_dir} does not exist."
        config_file = os.path.join(case_dir, 'config.yaml')

        # save original config
        _cfg = cfg

        # load config from experiment directory
        if GLOBAL_RANK == 0:
            print(f'Loading config from {config_file}')
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = Config(**{k: v for k, v in cfg.items() if k in Config.__annotations__})

        if _cfg.evaluate:
            cfg.evaluate = True
            cfg.train = False
        elif _cfg.restart:
            cfg.restart = True
            cfg.train = True

    if DISTRIBUTED:
        torch.distributed.barrier()

    main(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#
