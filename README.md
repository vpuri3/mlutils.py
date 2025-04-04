# `mlutils.py`

Pytorch template for ML projects based off the structre in [GeomLearning.py](https://github.com/vpuri3/GeomLearning.py/tree/master).

Features:
- Equiped for multi-gpu/ multi-node training with `torchrun`
- Extendable `Trainer` and `Callback` classes with checkpointing, restarts, and analysuis enabled
- Easy to get started with - just clone the repo and tune it for your application

## Installation and running
Download the repo
```sh
git clone https://github.com/vpuri3/mlutils.py.git
```

Install the environment
```sh
cd mlutls.py
chmod +x scripts/install.sh
./install.sh
```

Training: You can either run with vanilla python for single GPU training or with `torchrun` for multi-gpu/ multi-node training.
```sh
python -m project --exp_name case1 --dataset dummy --train true --epochs 100 ...
```
```sh
torchrun --nproc-per-node gpu -m project --exp_name case2 --dataset dummy --train true --epochs 100 ...
```
By default all cases are stored in `/out/`.
```sh
[vedantpu@eagle mlutils.py]:tree out/ -L 2
out/
├── case1
    ├── ckpt01
    ├── ...
    ├── ckpt10
    ├── config.yaml
    └── eval
└── case2
    ├── ckpt01
    ├── ...
    ├── ckpt10
    ├── config.yaml
    └── eval
```

Evaluation/Analysis: You can load a preexisting case for evaluation or analysis.
```sh
python -m project --config /out/case/config.yaml --train false --eval true
```
This will load the model, perform any analysis operations via the callback, and save the output to a new case directory.

See `python -m project --help` for details
```
(MLUtils) [vedantpu@eagle mlutils.py]:python -m project --help
usage: __main__.py [-h] [--config CONFIG] [--print_config[=flags]] [--train {true,false}] [--eval {true,false}]
                   [--restart_file RESTART_FILE] [--dataset DATASET] [--exp_name EXP_NAME] [--seed SEED]
                   [--model_type MODEL_TYPE] [--width WIDTH] [--num_layers NUM_LAYERS] [--num_heads NUM_HEADS]
                   [--mlp_ratio MLP_RATIO] [--num_slices NUM_SLICES] [--epochs EPOCHS]
                   [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--schedule SCHEDULE]
                   [--batch_size BATCH_SIZE]
...
```
