#
import os
import torch
from typing import Optional, Callable

__all__ = [
    'DummyDataset',
]

#======================================================================#
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        
        if not os.path.exists(self.root):
            print(f"Dataset root {self.root} does not exist. Creating it.")
            os.makedirs(self.root)

        if not os.path.exists(os.path.join(self.root, 'x.pt')):
            print(f"x.pt file not found in {self.root}. Creating it.")
            x_coords = torch.linspace(0, 1, 32)
            y_coords = torch.linspace(0, 1, 32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
            x = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            x = x.unsqueeze(0).repeat(1000, 1, 1)

            torch.save(x, os.path.join(self.root, 'x.pt'))

        self.x = torch.load(os.path.join(self.root, 'x.pt'), weights_only=True)

        if not os.path.exists(os.path.join(self.root, 'y.pt')):
            print(f"y.pt file not found in {self.root}. Creating it.")
            y = torch.sin(torch.pi * self.x[..., 0:1]) * torch.sin(torch.pi * self.x[..., 1:2])
            torch.save(y, os.path.join(self.root, 'y.pt'))
        
        self.y = torch.load(os.path.join(self.root, 'y.pt'), weights_only=True)
        
        self.transform.fit(self.x, self.y)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
#======================================================================#
