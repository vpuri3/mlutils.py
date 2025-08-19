#
import torch
import torch.distributed as dist
from tqdm import tqdm

import mlutils
import project

__all__ = [
    'RelL2Callback',
    'RelL2StatsFun',
]

#======================================================================#
class RelL2Callback(mlutils.Callback):
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        if trainer.GLOBAL_RANK == 0:
            train_rel_error = trainer.stat_vals['train_stats']['rel_l2']
            test_rel_error = trainer.stat_vals['test_stats']['rel_l2']

            print(f"Train/test relative error: {train_rel_error:.4f} / {test_rel_error:.4f}")

        return

#======================================================================#
class RelL2StatsFun:
    def __init__(self, x_normalizer=None, y_normalizer=None):
        self.x_normalizer = x_normalizer if x_normalizer is not None else project.IdentityNormalizer()
        self.y_normalizer = y_normalizer if y_normalizer is not None else project.IdentityNormalizer()

    def __call__(self, trainer, loader):

        if trainer.GLOBAL_RANK == 0:
            batch_iterator = tqdm(loader, desc="Evaluating (train/test) dataset", ncols=80)
        else:
            batch_iterator = loader

        model = trainer.model
        device = trainer.device

        loss = 0
        num_batches = 0
        lossfun = torch.nn.MSELoss()

        for batch in batch_iterator:
            x = batch[0].to(device)
            y = batch[1].to(device)

            with trainer.auto_cast:
                yh = model(x)
                l  = lossfun(yh, y).item()

            loss += l
            num_batches += len(x)

        if trainer.DDP:
            loss = torch.tensor(loss, device=device)
            num_batches = torch.tensor(num_batches, device=device)

            dist.all_reduce(num_batches, dist.ReduceOp.SUM)
            dist.all_reduce(loss, dist.ReduceOp.SUM)

            num_batches, loss = num_batches.item(), loss.item()

        loss = loss / num_batches
        rel_l2 = loss

        stats = dict(
            rel_l2=rel_l2,
        )

        return loss, stats

#======================================================================#
#