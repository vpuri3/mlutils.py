#
import torch

__all__ = [
    'NormalizeTransform',
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