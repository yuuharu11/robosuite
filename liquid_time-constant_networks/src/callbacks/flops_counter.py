import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from fvcore.nn import FlopCountAnalysis
import logging

log = logging.getLogger(__name__)

class FlopsCounterCallback(Callback):
    def __init__(self, enable: bool):
        self.enable = enable

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enable:
            return

        log.info("FLOPs計算を開始します...")
        model = pl_module.model.eval()
        device = pl_module.device

        # --- ダミー入力の準備（自動形状対応） ---
        dummy_input = None
        try:
            # datamoduleがなければtest_dataloadersから取得
            if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                batch = next(iter(trainer.datamodule.test_dataloader()))
            elif hasattr(trainer, "test_dataloaders") and trainer.test_dataloaders:
                batch = next(iter(trainer.test_dataloaders[0]))
            else:
                raise AttributeError("テスト用データローダが見つかりませんでした。")
            # batchがタプルの場合は最初の要素を使う
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            dummy_input = inputs[:1].to(device)
        except Exception as e:
            log.warning(f"入力形状の自動取得に失敗しました: {e}")
            log.warning("フォールバック: モデルのforwardの引数を確認してください。")
            return

        # --- FLOPsの計算 ---
        try:
            flops = FlopCountAnalysis(model, dummy_input)
            gflops = flops.total() / 1e9

            log.info("\n" + "="*60)
            log.info(" FLOPs Calculation Callback Report")
            log.info(f"  - Model: {model.__class__.__name__}")
            log.info(f"  - Input shape: {list(dummy_input.shape)}")
            log.info(f"  - Total GFLOPs: {gflops:.4f}")
            log.info("="*60 + "\n")

            pl_module.log("model_gflops", gflops, rank_zero_only=True)
        except Exception as e:
            log.error(f"FLOPsの計算に失敗しました: {e}")