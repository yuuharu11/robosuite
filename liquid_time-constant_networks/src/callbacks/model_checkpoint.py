"""
Model checkpoint callback wrapper
PyTorch LightningのModelCheckpointのラッパー
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as PLModelCheckpoint
from .base import BaseCallback


class ModelCheckpoint(BaseCallback):
    """
    Model checkpoint callback
    PyTorch LightningのModelCheckpointをラップ
    """
    
    def __init__(
        self,
        dirpath=None,
        filename=None,
        monitor="val_loss",
        verbose=False,
        save_last=True,
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=True,
        every_n_epochs=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.checkpoint = PLModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_epochs=every_n_epochs,
        )
    
    def setup(self, trainer, pl_module, stage=None):
        # Add the actual checkpoint callback to trainer
        if self.checkpoint not in trainer.callbacks:
            trainer.callbacks.append(self.checkpoint)
    
    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends"""
        pass
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends"""
        pass
