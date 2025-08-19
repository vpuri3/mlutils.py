#
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

__all__ = [
    'make_optimizer_adamw',
    'make_optimizer_lion',
    #
    'IdentityNormalizer',
    'UnitCubeNormalizer',
    'UnitGaussianNormalizer',
    'RelL2Loss',
]

#======================================================================#
def split_params(model):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        if name.endswith(".bias") or "LayerNorm" in name or "layernorm" in name or "embedding" in name.lower():
            no_decay.append(param)
        elif 'latent' in name:
            no_decay.append(param)
        elif 'cls_token' in name:
            no_decay.append(param)
        elif 'pos_embed' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay

#======================================================================#
def make_optimizer_adamw(model, lr, weight_decay=0.0, betas=None, eps=None, **kwargs):
    betas = betas if betas is not None else (0.9, 0.999)
    eps = eps if eps is not None else 1e-8

    decay, no_decay = split_params(model)

    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr, betas=betas, eps=eps)

    return optimizer

def make_optimizer_lion(model, lr, weight_decay=0.0, betas=None, eps=None, **kwargs):
    betas = betas if betas is not None else (0.9, 0.999)
    eps = eps if eps is not None else 1e-8

    decay, no_decay = split_params(model)
    
    optimizer = Lion([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr, betas=betas, eps=eps)

    return optimizer

#======================================================================#

class Lion(Optimizer):
    r"""
    Lion Optimizer (Chen et al., 2023):
    https://arxiv.org/abs/2302.06675

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        w_{t+1} = w_t - lr * sign(m_t)

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        betas (Tuple[float, float]): momentum coefficients (beta1, beta2). Note that beta2 is not used.
        weight_decay (float): optional weight decay (L2 penalty)
        eps (float): optional epsilon. Note that eps is not used.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay directly to weights
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # State (momentum) initialization
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Parameter update (sign of momentum)
                p.add_(exp_avg.sign(), alpha=-lr)

        return loss

#======================================================================#
class IdentityNormalizer():
    def __init__(self):
        pass
    
    def to(self, device):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x

#======================================================================#
class UnitCubeNormalizer():
    def __init__(self, X):
        mins = X.amin(dim=(0, 1), keepdim=True)
        maxs = X.amax(dim=(0, 1), keepdim=True)

        self.mins = mins
        self.maxs = maxs

    def to(self, device):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)

        return self

    def encode(self, x):
        x = (x - self.mins) / (self.maxs - self.mins)
        return x

    def decode(self, x):
        return x * (self.maxs - self.mins) + self.mins

#======================================================================#
class UnitGaussianNormalizer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

#======================================================================#
class RelL2Loss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        dim = tuple(range(1, pred.ndim))

        error = torch.sum((pred - target) ** 2, dim=dim).sqrt()
        target = torch.sum(target ** 2, dim=dim).sqrt()

        loss = torch.mean(error / target)
        return loss

#======================================================================#
#