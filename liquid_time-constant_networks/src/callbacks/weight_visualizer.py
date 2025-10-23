import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class WeightVisualizerCallback(Callback):
    """
    エポックの終わりに全ての重み行列とマスクを自動で可視化しWandBに記録するコールバック
    """
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        print(f"\n[Visualizer] Logging weights and masks on train end.")

        # named_parametersで全パラメータを取得
        for name, param in pl_module.model.named_parameters():
            # パラメータが2次元以上なら可視化
            if param.ndim < 2:
                continue

            weights = param.data.clone().cpu().numpy()

            # Figure作成
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'{name} | Training End', fontsize=16)

            sns.heatmap(weights.reshape(weights.shape[0], -1), ax=axes[0], cmap='viridis', cbar=False)
            axes[0].set_title('Weights')
            axes[0].set_xlabel('Flattened')
            axes[0].set_ylabel('Rows')

            axes[1].hist(weights.flatten(), bins=50, color='blue', alpha=0.7)
            axes[1].set_title('Weights Histogram')
            axes[1].set_xlabel('Weight Value')
            axes[1].set_ylabel('Frequency')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            trainer.logger.experiment.log({
                f"weights/{name}": wandb.Image(fig),
                "trainer/global_step": trainer.global_step
            })
            plt.close(fig)