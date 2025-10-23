import torch
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pytorch_lightning.callbacks import Callback

class WeightChangeVisualizerCallback(Callback):
    """
    タスクごとに重み行列の変化を可視化・記録するコールバック
    """
    def __init__(self, enable=False):
        self.prev_weights = {}
        self.enable = enable

    def on_train_start(self, trainer, pl_module):
        if not self.enable:
            return  
        # 学習開始時の重みを保存
        self.prev_weights = {name: param.data.clone().cpu() for name, param in pl_module.model.named_parameters()}

    def on_train_end(self, trainer, pl_module):
        if not self.enable:
            return
        print("[Visualizer] Logging weight changes on train end.")
        for name, param in pl_module.model.named_parameters():
            if param.ndim < 2:
                continue
            weights = param.data.clone().cpu().numpy()
            prev = self.prev_weights.get(name)
            if prev is not None:
                diff = weights - prev.numpy()
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'{name} | Weight Change', fontsize=16)
                sns.heatmap(weights, ax=axes[0], cmap='viridis')
                axes[0].set_title('Current Weights')
                sns.heatmap(diff, ax=axes[1], cmap='coolwarm')
                axes[1].set_title('Weight Change (Δ)')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                trainer.logger.experiment.log({
                    f"weight_change/{name}": wandb.Image(fig),
                    "trainer/global_step": trainer.global_step
                })
                plt.close(fig)