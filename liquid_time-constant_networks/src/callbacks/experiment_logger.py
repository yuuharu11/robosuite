import csv
import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from rich import print
import re

class TrainingMonitor(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()
        self.training_peak_memory = []
        self.training_reserved_mem = []
        if pl_module.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.device.type == 'cuda':
            if batch_idx > 0: 
                self.training_peak_memory.append(torch.cuda.max_memory_allocated(pl_module.device))
                self.training_reserved_mem.append(torch.cuda.max_memory_reserved(pl_module.device))
                torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.perf_counter() - self.epoch_start_time
        pl_module.log("training/epoch_duration_sec", epoch_duration, on_step=False, on_epoch=True, logger=True)
        
        if self.training_peak_memory and pl_module.device.type == 'cuda':
            avg_peak_memory_mb = np.mean(self.training_peak_memory) / (1024**2)
            avg_reserved_memory_mb = np.mean(self.training_reserved_mem) / (1024**2)
            pl_module.log("training/avg_peak_mb", avg_peak_memory_mb, on_step=False, on_epoch=True, logger=True)
            pl_module.log("training/avg_reserved_mb", avg_reserved_memory_mb, on_step=False, on_epoch=True, logger=True)
            print(f"\n[cyan]TrainingMonitor: Avg Peak Memory this Epoch: {avg_peak_memory_mb:.2f} MB, Duration: {epoch_duration:.2f} sec[/cyan]")
            print(f"[cyan]TrainingMonitor: Avg Reserved Memory this Epoch: {avg_reserved_memory_mb:.2f} MB[/cyan]")

class InferenceMonitor(Callback):
    def on_test_epoch_start(self, trainer, pl_module):
        self.latencies = []
        self.inference_peak_memory = []
        self.inference_reserved_mem = []
        print("\n[yellow]InferenceMonitor: Starting measurement...[/yellow]")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.start_time = time.perf_counter()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.device.type == 'cuda':
            if batch_idx > 0: 
                torch.cuda.synchronize()
                self.inference_peak_memory.append(torch.cuda.max_memory_allocated(pl_module.device))
                self.inference_reserved_mem.append(torch.cuda.max_memory_reserved(pl_module.device))
        
        latency = time.perf_counter() - self.start_time
        self.latencies.append(latency)

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.latencies: return
        stable_latencies = self.latencies[1:] if len(self.latencies) > 1 else self.latencies
        avg_stable_latency_ms = np.mean(stable_latencies) * 1000
        p95_latency_ms = np.percentile(stable_latencies, 95) * 1000
        
        pl_module.log("inference/avg_latency_ms", avg_stable_latency_ms, logger=True)
        pl_module.log("inference/p95_latency_ms", p95_latency_ms, logger=True)

        avg_peak_memory_mb = 0.0
        if self.inference_peak_memory and pl_module.device.type == 'cuda':
            avg_peak_memory_mb = np.mean(self.inference_peak_memory) / (1024**2)
            pl_module.log("inference/avg_peak_mb", avg_peak_memory_mb, logger=True)

        avg_reserved_memory_mb = 0.0
        if self.inference_reserved_mem and pl_module.device.type == 'cuda':
            avg_reserved_memory_mb = np.mean(self.inference_reserved_mem) / (1024**2)
            pl_module.log("inference/avg_reserved_mb", avg_reserved_memory_mb, logger=True)
        print(f"\n[green]InferenceMonitor: Avg Latency: {avg_stable_latency_ms:.2f} ms, P95 Latency: {p95_latency_ms:.2f} ms[/green]")
        print(f"[green]InferenceMonitor: Avg Peak Memory: {avg_peak_memory_mb:.2f} MB, Avg Reserved Memory: {avg_reserved_memory_mb:.2f} MB[/green]")

class CSVSummaryCallback(Callback):
    def __init__(self, enable, output_file="results/summary.csv"):
        super().__init__()
        self.enable = enable
        self.output_file = output_file
        self.results_cache = {}
        self.has_written_this_run = False
        self.headers = [
            "phase_type", "noise_level", "task",  "テスト精度 (Test Acc)", "平均レイテンシ (ms/バッチ)",
            "p95 レイテンシ (ms/バッチ)", "推論時 Memory Allocated [MB]", "推論時 Memory Reserved [MB]", 
            "学習時間/epoch", "訓練時 Memory Allocated [MB]", "訓練時 Memory Reserved [MB]", "検証精度 (Val Acc)", "チェックポイントパス"
        ]

    def _get_metric(self, metrics_dict, key, default=-1.0):
        value = metrics_dict.get(key, default)
        return value.item() if isinstance(value, torch.Tensor) else value

    def _capture_hparams(self, trainer: Trainer, pl_module):
        if not self.enable:
            return
        hparams = pl_module.hparams
        """
        self.results_cache["データセット"] = hparams.dataset._name_
        self.results_cache["batch"] = hparams.loader.batch_size
        self.results_cache["model.n_layers"] = hparams.model.get("n_layers", "N/A")
        self.results_cache["model.d_model"] = hparams.model.get("d_model", "N/A")
        #units_list = hparams.model.layer.get("units", [])
        #self.results_cache["units"] = next((item.get("units") for item in units_list if "units" in item), "N/A")
        #self.results_cache["output_units"] = next((item.get("output_units") for item in units_list if "output_units" in item), "N/A")
        #self.results_cache["ode_solver_unfolds"] = hparams.model.layer.get("ode_unfolds", "N/A")
        """

    def on_train_end(self, trainer: Trainer, pl_module):
        if self.has_written_this_run or not self.enable:
            return
        #self._capture_hparams(trainer, pl_module)
        metrics = trainer.logged_metrics
        self.results_cache["検証精度 (Val Acc)"] = self._get_metric(metrics, "final/val/accuracy_epoch")
        self.results_cache["学習時間/epoch"] = self._get_metric(metrics, "training/epoch_duration_sec")
        self.results_cache["訓練時 Memory Allocated [MB]"] = self._get_metric(metrics, "training/avg_peak_mb")
        self.results_cache["訓練時 Memory Reserved [MB]"] = self._get_metric(metrics, "training/avg_reserved_mb")
        
        print("\n[bold cyan]CSVSummaryCallback: 訓練結果をキャプチャしました。[/bold cyan]")

    def on_test_start(self, trainer: Trainer, pl_module):
        self.has_written_this_run = False

    def on_test_end(self, trainer: Trainer, pl_module):
        if self.has_written_this_run:
            return
        test_only_flag = getattr(pl_module.hparams.train, "test_only", False)
        self.results_cache["phase_type"] = "inference" if test_only_flag else "training"

        metrics = trainer.callback_metrics
        test_metric_key = f"final/test/{pl_module.hparams.task.get('metric', 'accuracy')}"
        self.results_cache["noise_level"] = pl_module.hparams.dataset.get("noise_level", -1)
        self.results_cache["task"] = pl_module.hparams.dataset.get("task_id", -1)
        self.results_cache["テスト精度 (Test Acc)"] = self._get_metric(metrics, test_metric_key)
        self.results_cache["平均レイテンシ (ms/バッチ)"] = self._get_metric(metrics, "inference/avg_latency_ms")
        self.results_cache["p95 レイテンシ (ms/バッチ)"] = self._get_metric(metrics, "inference/p95_latency_ms")
        self.results_cache["推論時 Memory Allocated [MB]"] = self._get_metric(metrics, "inference/avg_peak_mb")
        self.results_cache["推論時 Memory Reserved [MB]"] = self._get_metric(metrics, "inference/avg_reserved_mb")

        # その他情報を取得
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_path = trainer.ckpt_path or "N/A"
                self.results_cache["チェックポイントパス"] = cb.best_model_path or ckpt_path
        
        # CSVファイルに書き出す
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        file_exists = os.path.isfile(self.output_file)
        with open(self.output_file, "a", newline="", encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            if not file_exists:
                writer.writeheader()
            row_data = {h: self.results_cache.get(h, "") for h in self.headers}
            writer.writerow(row_data)
        
        self.has_written_this_run = True 
        self.results_cache = {} # 次の実験のためにキャッシュをクリア
        print(f"\n[bold magenta]📊 全ての実験結果を {self.output_file} に記録しました。[/bold magenta]")