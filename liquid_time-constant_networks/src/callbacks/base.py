"""
Base callback class
"""

import pytorch_lightning as pl
from abc import ABC


class BaseCallback(pl.Callback, ABC):
    """
    Base callback class for custom callbacks
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
    
    def setup(self, trainer, pl_module, stage=None):
        """Called when fit, validate, test, predict, or tune begins"""
        pass
    
    def teardown(self, trainer, pl_module, stage=None):
        """Called when fit, validate, test, predict, or tune ends"""
        pass
    
    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins"""
        pass
    
    def on_fit_end(self, trainer, pl_module):
        """Called when fit ends"""
        pass
    
    def on_train_start(self, trainer, pl_module):
        """Called when the train begins"""
        pass
    
    def on_train_end(self, trainer, pl_module):
        """Called when the train ends"""
        pass
    
    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins"""
        pass
    
    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends"""
        pass
    
    def on_epoch_start(self, trainer, pl_module):
        """Called when either of train/val/test epoch begins"""
        pass
    
    def on_epoch_end(self, trainer, pl_module):
        """Called when either of train/val/test epoch ends"""
        pass
