#
import torch
import mlutils

__all__ = [
    'RelErrorCallback',
]

#======================================================================#
class RelErrorCallback(mlutils.Callback):
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        for dataset in [trainer._data, trainer.data_]:
            transform = dataset.dataset.transform
            x, y = torch.utils.data.default_collate([dataset[i] for i in range(len(dataset))])
            x_norm, y_norm = transform(x, y)

            # Get model predictions
            with torch.no_grad():
                y_pred_norm = trainer.model(x_norm.to(trainer.device)).to('cpu')
                y_pred = transform.unnormalize_y(y_pred_norm)
            
            # Compute relative error
            rel_error = torch.mean(torch.square(y - y_pred) / (torch.square(y) + 1e-8)).sqrt()
            
            if trainer.GLOBAL_RANK == 0:
                print(f"Relative error: {rel_error.item():.4f}")
            
        return

#======================================================================#
#
