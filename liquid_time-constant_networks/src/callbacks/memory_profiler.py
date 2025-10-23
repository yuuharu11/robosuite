import os
from typing import Optional
from pytorch_lightning import Callback, Trainer, LightningModule
from hydra.utils import get_original_cwd
import torch
from torch import profiler

class ProfilerCallback(Callback):
    """
    指定した start_step から profile_steps 分だけ torch.profiler を有効化して trace を出力する。
    - 出力先: <get_original_cwd()>/logs/profiler (既定)
    - profile_memory はデフォルト False（必要なときだけ True に）
    - enable_callback を False にすると Callback は no-op になります（YAML から渡せます）。
    """

    def __init__(
        self,
        enable_callback: bool = False,
        start_step: int = 10,
        profile_steps: int = 20,
        output_dir: Optional[str] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
    ):
        super().__init__()
        # YAML 等から enable_callback を受け取り動作を切り替えられるようにする
        self.enabled = bool(enable_callback)
        self.start_step = int(start_step)
        self.profile_steps = int(profile_steps)
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)

        self._active = False
        self._prof = None

        # 出力先の作成は有効時のみ行う（無効ならログディレクトリを作らない）
        self.output_dir = None
        if self.enabled:
            self.output_dir = output_dir or os.path.join(get_original_cwd(), "logs", "profiler")
            os.makedirs(self.output_dir, exist_ok=True)

    def _start_profiler(self):
        if not self.enabled:
            return
        # schedule を使って自動停止も可能にするが、ここでは手動で __exit__ を呼ぶ
        self._prof = profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=0, active=self.profile_steps, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(self.output_dir) if self.output_dir is not None else None,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self._prof.__enter__()
        self._active = True

    def _stop_profiler(self):
        if not self.enabled:
            return
        try:
            if self._prof is not None:
                self._prof.__exit__(None, None, None)
        finally:
            self._prof = None
            self._active = False

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, dataloader_idx=0):
        if not self.enabled:
            return
        # trainer.global_step は optimizer.step の後に増える場合があるが、ここではこの値でトリガする
        gs = int(trainer.global_step)
        if not self._active and gs >= self.start_step:
            # 開始条件満たしたら開始
            self._start_profiler()

        if self._active and self._prof is not None:
            # profiler の内部スケジュールを進める
            try:
                self._prof.step()
            except Exception:
                pass

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.enabled:
            return
        if self._active:
            # 終了判定
            gs = int(trainer.global_step)
            # trainer.global_step が step 処理後に増える実装差があるため余裕を持たせる
            if gs >= self.start_step + self.profile_steps:
                self._stop_profiler()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.enabled:
            return
        # 万が一残っていたら停止して出力を確実にフラッシュ
        if self._active:
            self._stop_profiler()