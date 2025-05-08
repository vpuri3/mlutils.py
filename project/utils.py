#
import torch

__all__ = [
    'NormalizeTransform',
    'IdentityNormalizer',
    'UnitCubeNormalizer',
    'UnitGaussianNormalizer',
    'RelL2Loss',
]

#======================================================================#

class NormalizeTransform:
    def __init__(self):
        pass

    def fit(self, x, y):
        # Compute normalization parameters for x (unit cube)
        self.x_min = x.amin(dim=(0, 1), keepdim=True)
        self.x_max = x.amax(dim=(0, 1), keepdim=True)
        # Compute normalization parameters for y (zero mean, unit variance)
        self.y_mean = y.mean(dim=(0, 1), keepdim=True)
        self.y_std = y.std(dim=(0, 1), keepdim=True)

    def __call__(self, x, y):
        # Normalize x to unit cube [0, 1]
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)

        # Normalize y to zero mean and unit variance
        y_norm = (y - self.y_mean) / self.y_std

        return x_norm, y_norm

    def unnormalize_x(self, x_norm):
        # Unnormalize x from unit cube
        x = x_norm * (self.x_max - self.x_min) + self.x_min

        return x

    def unnormalize_y(self, y_norm):
        # Unnormalize y from zero mean, unit variance
        y = y_norm * self.y_std + self.y_mean

        return y

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
        xmin = X[:,:,0].min().item()
        ymin = X[:,:,1].min().item()

        xmax = X[:,:,0].max().item()
        ymax = X[:,:,1].max().item()

        self.min = torch.tensor([xmin, ymin])
        self.max = torch.tensor([xmax, ymax])

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)

        return self

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        return x * (self.max - self.min) + self.min

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