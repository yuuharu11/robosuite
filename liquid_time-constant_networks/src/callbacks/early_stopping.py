"""
Early stopping callback wrapper
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping as PLEarlyStopping
from .base import BaseCallback


class EarlyStopping(BaseCallback):
    """
    Early stopping callback
    PyTorch LightningのEarlyStoppingをラップ
    """
    
    def __init__(
        self,
        monitor="val_loss",
        min_delta=0.0,
        patience=3,
        verbose=False,
        mode="min",
        strict=True,
        check_finite=True,
        stopping_threshold=None,
        divergence_threshold=None,
        check_on_train_epoch_end=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.early_stopping = PLEarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
    
    def setup(self, trainer, pl_module, stage=None):
        # Add the actual early stopping callback to trainer
        if self.early_stopping not in trainer.callbacks:
            trainer.callbacks.append(self.early_stopping)
    
    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends"""
        pass
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends"""  
        pass
