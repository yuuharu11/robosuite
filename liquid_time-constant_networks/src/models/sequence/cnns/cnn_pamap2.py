# src/models/sequence/cnns/cnn.py

import torch
import torch.nn as nn
from src.utils import registry

class CNN_PAMAP2(nn.Module):
    """
    A simple 1D CNN model specifically tailored for the PAMAP2 dataset.
    """
    def __init__(
        self,
        d_input: int,  # PAMAP2では40
        d_output: int, # PAMAP2では12
        **kwargs,      # 他の不要な引数を吸収
    ):
        super().__init__()
        
        # --- 畳み込みブロック1 ---
        self.conv1 = nn.Conv1d(
            in_channels=d_input, 
            out_channels=64, 
            kernel_size=5, 
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- 畳み込みブロック2 ---
        self.conv2 = nn.Conv1d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=5, 
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- 全結合層（分類器） ---
        # PAMAP2のシーケンス長 512 -> pool1 -> 256 -> pool2 -> 128
        # この計算を自動化するためにAdaptive Poolingを使用するのが最も堅牢
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, d_output) # 最終チャンネル数(128) -> 出力クラス数

    def forward(self, x: torch.Tensor, state=None, **kwargs) -> torch.Tensor:
        # 入力形状: (Batch, Length, Channels)
        # Conv1dが期待する形状: (Batch, Channels, Length)
        x = x.permute(0, 2, 1)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # -> (Batch, 128, 128)
        
        x = self.final_pool(x) # -> (Batch, 128, 1)
        x = self.flatten(x)    # -> (Batch, 128)
        x = self.fc(x)         # -> (Batch, d_output)
        
        return x, None

    # train.pyのstate処理ロジックと互換性を保つために必要
    def default_state(self, *args, **kwargs):
        return None