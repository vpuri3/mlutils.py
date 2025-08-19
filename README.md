# `mlutils.py`

A lightweight and extensible PyTorch machine learning framework designed for rapid prototyping and scalable training of deep learning models.

## 🚀 Features

- **Distributed Training**: Built-in support for multi-GPU and multi-node training with `torchrun`
- **Mixed Precision**: Automatic mixed precision training with gradient scaling
- **Flexible Architecture**: Easy-to-extend modular design for models, datasets, and callbacks
- **Comprehensive Training**: Automated checkpointing, learning rate scheduling, and statistics tracking
- **Command-line Interface**: YAML configuration with command-line overrides
- **Production Ready**: Gradient clipping, memory monitoring, and robust error handling

## 📁 Project Structure

```
mlutils.py/
├── mlutils/              # Core ML framework
│   ├── trainer.py       # Main Trainer class with distributed training support
│   ├── callbacks.py     # Callback system for training hooks
│   ├── schedule.py      # Learning rate schedulers
│   └── utils.py         # Utilities (device selection, parameter counting, etc.)
├── project/              # Example project implementation
│   ├── models/          # Model implementations
│   │   └── transformer.py  # Example transformer model
│   ├── datasets/        # Dataset implementations
│   │   ├── dummy.py     # Example dummy dataset
│   │   └── utils.py     # Dataset loading utilities
│   ├── callbacks.py     # Project-specific callbacks
│   ├── utils.py         # Optimizers (AdamW, Lion), normalizers, loss functions
│   └── __main__.py      # Training script with CLI
├── scripts/             # Installation and utility scripts
└── pyproject.toml       # Project dependencies
```

## 🛠️ Installation

### Prerequisites
- Python >= 3.11
- PyTorch (automatically installed)

### Quick Install

```bash
git clone <repository-url>
cd mlutils.py
pip install -e .
```

### Development Install

For development with additional tools:

```bash
git clone <repository-url>
cd mlutils.py
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Training a Model

Single GPU training:
```bash
python -m project --train true --dataset dummy --exp_name my_experiment --epochs 50
```

Multi-GPU training:
```bash
torchrun --nproc-per-node 2 -m project --train true --dataset dummy --exp_name my_experiment --epochs 50
```

### Evaluating a Model

```bash
python -m project --evaluate true --exp_name my_experiment
```

### Resuming from Checkpoint

```bash
python -m project --restart true --exp_name my_experiment
```

## ⚙️ Configuration

The framework uses YAML configuration files with command-line overrides. Key configuration options:

```bash
python -m project --help
```

### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 0.0)
- `--optimizer`: Optimizer choice (adamw, lion)
- `--mixed_precision`: Enable mixed precision training (default: true)

### Model Parameters
- `--model_type`: Model type (0: Transformer)
- `--channel_dim`: Model hidden dimension (default: 64)
- `--num_blocks`: Number of transformer blocks (default: 4)
- `--num_heads`: Number of attention heads (default: 8)
- `--mlp_ratio`: MLP expansion ratio (default: 4.0)

## 🏗️ Core Components

### Trainer Class

The `mlutils.Trainer` class provides:

- **Distributed Training**: Automatic handling of multi-GPU training with DDP
- **Mixed Precision**: Built-in support for FP16 training with gradient scaling
- **Checkpointing**: Automatic model and optimizer state saving/loading
- **Callbacks**: Extensible hook system for custom training logic
- **Statistics**: Training/validation loss tracking and custom metrics

### Example Usage

```python
import mlutils
import project

# Load data
train_data, test_data, metadata = project.load_dataset('dummy', 'data/')

# Create model
model = project.Transformer(
    in_dim=metadata['in_dim'], 
    out_dim=metadata['out_dim'],
    channel_dim=64,
    num_blocks=4,
    num_heads=8
)

# Create trainer
trainer = mlutils.Trainer(
    model=model,
    _data=train_data,
    data_=test_data,
    epochs=100,
    lr=1e-3,
    mixed_precision=True
)

# Train
trainer.train()
```

### Adding Custom Models

Create new models in `project/models/`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.layers = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.layers(x)
```

Register in `project/models/__init__.py`:
```python
from .my_model import MyModel
```

### Adding Custom Datasets

Create new datasets in `project/datasets/`:

```python
import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, root):
        # Load your data
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

Register in `project/datasets/utils.py`:
```python
def load_dataset(dataset_name, DATADIR_BASE):
    if dataset_name == 'my_dataset':
        # Load and return train, test, metadata
        pass
```

## 📊 Results and Monitoring

Training results are automatically saved to `out/<exp_name>/`:

```
out/
└── my_experiment/
    ├── config.yaml          # Experiment configuration
    ├── ckpt01/              # Checkpoints (model + optimizer state)
    ├── ckpt02/
    ├── ...
    ├── losses.png           # Training/validation loss plots
    ├── learning_rate.png    # Learning rate schedule
    ├── grad_norm.png        # Gradient norm tracking
    └── rel_error.json       # Final evaluation metrics
```

## 🔧 Advanced Features

### Custom Optimizers

The framework includes implementations of:
- **AdamW**: With automatic parameter group separation (decay vs no-decay)
- **Lion**: Memory-efficient optimizer with sign-based updates

### Data Normalizers

Built-in normalizers for data preprocessing:
- `IdentityNormalizer`: No normalization
- `UnitCubeNormalizer`: Scale to [0,1] range
- `UnitGaussianNormalizer`: Zero mean, unit variance

### Callback System

Extensible callback system for training hooks:

```python
class MyCallback(mlutils.Callback):
    def __call__(self, trainer, **kwargs):
        # Custom logic during training
        pass

trainer.add_callback('epoch_end', MyCallback())
```

### Learning Rate Schedules

Supported schedules:
- `OneCycleLR`: One-cycle learning rate policy
- `CosineAnnealingLR`: Cosine annealing
- `CosineAnnealingWarmRestarts`: Cosine annealing with restarts
- `ConstantLR`: Constant learning rate

## 🧪 Example: Dummy Dataset

The included dummy dataset demonstrates the framework:
- **Input**: 2D coordinates (x, y) on a 32×32 grid
- **Output**: sin(πx) × sin(πy) function values
- **Task**: Function approximation with a transformer model

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model/dataset/feature following the existing patterns
4. Submit a pull request

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{mlutils2025,
  title={mlutils.py: A Lightweight PyTorch ML Framework},
  author={Vedant Puri},
  year={2025},
  url={https://github.com/vpuri3/mlutils.py}
}
```

---

**Note**: This framework is designed to be simple yet powerful. It provides the essential components needed for most ML projects while remaining easy to understand and extend.