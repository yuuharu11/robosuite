import numpy as np
import torch
import pytorch_lightning as pl
from rich import print
from .base import BaseCallback

def format_bytes(size):
    """Converts bytes to a human-readable format (KB, MB, GB)."""
    if size == 0:
        return "0 B"
    # The units for byte sizes
    power_name = ("B", "KB", "MB", "GB", "TB")
    # Determine the appropriate unit
    i = int(np.floor(np.log(size) / np.log(1024)))
    # Format the number
    p = np.power(1024, i)
    s = round(size / p, 2)
    return f"{s} {power_name[i]}"

class MemoryMonitor(BaseCallback):
    """
    Measures GPU memory consumption during inference in PyTorch Lightning. 📊
    """

    def on_test_epoch_start(self, trainer, pl_module):
        """Called once just before the test starts"""
        # Ensure a CUDA device is available
        if pl_module.device.type != 'cuda':
            print("\n[bold red]MemoryMonitor: A CUDA device is required to monitor memory usage.[/bold red]")
            self.disabled = True
            return
            
        self.disabled = False
        # Initialize lists to store measurement results
        self.batch_peak_memory = []
        self.total_samples = 0
        print("\n[yellow]MemoryMonitor: Starting measurement...[/yellow]")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called just before each batch processing starts"""
        if self.disabled:
            return
        
        # Reset the peak memory statistics for the current device
        torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called immediately after each batch processing ends"""
        if self.disabled:
            return
            
        # Get the peak memory allocated on the current device since the last reset
        peak_memory = torch.cuda.max_memory_allocated(pl_module.device)
        self.batch_peak_memory.append(peak_memory)
        
        # Record the number of processed samples
        self.total_samples += batch[0].size(0)

    def on_test_epoch_end(self, trainer, pl_module):
        """Called once after all test batches are completed"""
        if self.disabled:
            return

        print("\n" + "="*60)
        print("[bold green]Inference Memory Consumption Report[/bold green]")
        print("="*60)
        
        if not self.batch_peak_memory:
            print("[yellow]No batches were processed. Cannot generate report.[/yellow]")
            return

        # The first batch can have higher memory usage due to caching, so we can analyze stable usage
        stable_memory = self.batch_peak_memory[1:] if len(self.batch_peak_memory) > 1 else self.batch_peak_memory
        
        # Calculate statistical information
        avg_stable_peak_memory = np.mean(stable_memory)
        max_stable_peak_memory = np.max(stable_memory)
        p95_peak_memory = np.percentile(stable_memory, 95)

        print(f"Processed samples: {self.total_samples} items")
        print(f"Processed batches: {len(self.batch_peak_memory)} batches")
        print("-" * 30)
        print(f"Average peak memory (stable): [bold green]{format_bytes(avg_stable_peak_memory)}[/bold green]")
        print(f"Max peak memory (stable): [bold red]{format_bytes(max_stable_peak_memory)}[/bold red]")
        print(f"95th percentile peak memory (stable): {format_bytes(p95_peak_memory)}")
        print("="*60 + "\n")
        pl_module.log("memory/avg_peak_mb_stable", avg_stable_peak_memory / (1024**2), on_step=False, on_epoch=True)
        pl_module.log("memory/max_peak_mb_stable", max_stable_peak_memory / (1024**2), on_step=False, on_epoch=True)
        pl_module.log("memory/p95_peak_mb_stable", p95_peak_memory / (1024**2), on_step=False, on_epoch=True)