#
import os
import torch

from .dummy import DummyDataset
from project import UnitCubeNormalizer, UnitGaussianNormalizer

__all__ = [
    'load_dataset',
]

#======================================================================#
def load_dataset(
        dataset_name: str,
        DATADIR_BASE: str,
    ):
    """Load a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        tuple: (train_dataset, test_dataset, metadata) containing the loaded datasets and optional metadata dictionary
    """
    if dataset_name == 'dummy':
        #----------------------------------------------------------------#
        # Dummy Dataset
        #----------------------------------------------------------------#

        DATADIR = os.path.join(DATADIR_BASE, 'DummyDataset')
        dataset = DummyDataset(DATADIR)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.80, 0.20])
        metadata = dict(
            in_dim=2,
            out_dim=1,
            x_normalizer=UnitCubeNormalizer(train_dataset.dataset.x),
            y_normalizer=UnitGaussianNormalizer(train_dataset.dataset.y),
        )

        return train_dataset, test_dataset, metadata

    else:
        #----------------------------------------------------------------# 
        # Dataset not found
        #----------------------------------------------------------------# 
        raise ValueError(f"Dataset {dataset_name} not found.") 

#======================================================================#