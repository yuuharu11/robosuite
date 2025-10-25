optimizer = {
    "adamw": "torch.optim.AdamW",
}

scheduler = {
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
}

model = {
    "sequence": "src.models.sequence.SequenceModel",
    "cnn_pamap2": "src.models.sequence.cnns.cnn_pamap2.CNN_PAMAP2",
    "cnn_har": "src.models.sequence.cnns.cnn_har.CNN_UCI",
    "pnn": "src.models.sequence.pnn.PNN",
    "cfc": "src.models.ncps.cfc.CfC",
}

wirings = {
    "wiring": "src.models.wirings.Wiring",
    "fully_connected": "src.models.wirings.FullyConnected",
    "ncp": "src.models.wirings.NCP",
    "auto_ncp": "src.models.wirings.AutoNCP",
    "random": "src.models.wirings.Random",
}

layer = {
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "rnn_original": "src.models.sequence.rnns.rnn_original.RNN",
    "lstm": "src.models.baseline.lstm.TorchLSTM",
    "ncps_ltc": "src.models.ncps.ltc.LTC",
    "cfc": "src.models.ncps.cfc.CfC",
    "wired_cfc": "src.models.ncps.wired_cfc.WiredCfC",
    "cnn": "src.models.sequence.cnns.cnn.CNN",
}

cell = {
    "rnn": "src.models.sequence.rnns.cells.rnn.RNNCell",
    "ltc": "src.models.ncps.cells.ltc.LTCCell",
    "ncps_ltc": "src.models.ncps.cells.ltc_cell.LTCCell",
    "cfc": "src.models.ncps.cells.cfc_cell.CfCCell",
    "wired_cfc": "src.models.ncps.cells.wired_cfc_cell.WiredCfCCell",
}

callbacks = {
    "score": "src.callbacks.score.Score",
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
    # my callbacks
    "memory_monitor": "src.callbacks.memory_monitor.MemoryMonitor",
    "latency_monitor": "src.callbacks.latency_monitor.LatencyMonitor",
    "experiment_logger": "src.callbacks.experiment_logger.CSVSummaryCallback",
    "memory_profiler": "src.callbacks.memory_profiler.ProfilerCallback",
    "weight_visualizer": "src.callbacks.weight_visualizer.WeightVisualizerCallback",
    "weight_change_visualizer": "src.callbacks.weight_change_visualizer.WeightChangeVisualizerCallback",
    "flops_counter": "src.callbacks.flops_counter.FlopsCounterCallback",
}
