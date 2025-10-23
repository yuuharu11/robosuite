import time
import numpy as np
import torch
import pytorch_lightning as pl
from rich import print
from .base import BaseCallback

class LatencyMonitor(BaseCallback):
    """
    to measure latency of inference in PyTorch Lightning.
    """

    def on_test_epoch_start(self, trainer, pl_module):
        """Called once just before the test starts"""
        # Initialize lists to store measurement results
        self.batch_latencies = []
        self.total_samples = 0
        print("\n[yellow]LatencyMonitor: Starting measurement...[/yellow]")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called just before each batch processing starts"""
        # GPU processing is asynchronous (CPU proceeds to next instruction without waiting for completion),
        # so we wait for all previous processing to complete for accurate timing measurement.
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize()
        # Record the processing start time
        self.start_time = time.perf_counter()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called immediately after each batch processing ends"""
        # Similarly, wait for all GPU processing to complete before recording end time.
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate processing time and add to list
        latency = end_time - self.start_time
        self.batch_latencies.append(latency)
        # Record the number of processed samples
        self.total_samples += batch[0].size(0)

    def on_test_epoch_end(self, trainer, pl_module):
        """Called once after all test batches are completed"""
        print("\n" + "="*60)
        print("[bold green]Inference Latency Measurement Report[/bold green]")
        print("="*60)
        
        if not self.batch_latencies:
            print("[yellow]No batches were processed. Cannot generate report.[/yellow]")
            return

        # The first batch tends to be slower due to data loading initialization etc.,
        # so calculating "stable" latency from the 2nd batch onwards gives more realistic values.
        stable_latencies = self.batch_latencies[1:] if len(self.batch_latencies) > 1 else self.batch_latencies
        
        # Calculate statistical information
        avg_stable_batch_latency = np.mean(stable_latencies)
        p95_latency = np.percentile(stable_latencies, 95)
        throughput = self.total_samples / sum(self.batch_latencies) if sum(self.batch_latencies) > 0 else 0

        print(f"Processed samples: {self.total_samples} items")
        print(f"Processed batches: {len(self.batch_latencies)} batches")
        print(f"Throughput (samples/sec): [bold blue]{throughput:.2f}[/bold blue]")
        print("-" * 30)
        print(f"Average latency (stable): [bold green]{avg_stable_batch_latency:.6f} sec/batch[/bold green]")
        print(f"95th percentile latency (stable): {p95_latency:.6f} sec/batch")
        print("="*60 + "\n")

        pl_module.log("latency/avg_ms_stable", avg_stable_batch_latency * 1000, on_step=False, on_epoch=True)
        pl_module.log("latency/p95_ms_stable", p95_latency * 1000, on_step=False, on_epoch=True)