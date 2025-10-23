"""
Learning rate monitor callback wrapper
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor as PLLearningRateMonitor
from .base import BaseCallback


class LearningRateMonitor(BaseCallback):
    """
    Learning rate monitor callback
    PyTorch LightningのLearningRateMonitorをラップ
    """
    
    def __init__(
        self,
        logging_interval="epoch",  # "step" or "epoch"
        log_momentum=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.lr_monitor = PLLearningRateMonitor(
            logging_interval=logging_interval,
            log_momentum=log_momentum,
        )
    
    def setup(self, trainer, pl_module, stage=None):
        # Add the actual LR monitor callback to trainer
        if self.lr_monitor not in trainer.callbacks:
            trainer.callbacks.append(self.lr_monitor)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the train batch ends"""
        pass
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends"""
        pass
