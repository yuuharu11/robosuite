import torch
import torch.nn as nn

class CNN_UCI(nn.Module):
    """
    A simple 1D CNN model for time-series classification.
    (Corrected Version with Global Pooling and Dropout)
    """
    def __init__(self, d_model: int, d_output: int, dropout: float = 0.5):
        """
        Initializes the corrected CNN model.

        Args:
            d_model (int): The number of input features (channels). 
                              For UCI-HAR, this is 9.
            d_output (int): The number of output classes. 
                               For UCI-HAR, this is 6.
            dropout (float): Dropout rate for regularization.
        """
        super(CNN_UCI, self).__init__()
        
        # 畳み込みブロック1
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, 
                out_channels=64, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 畳み込みブロック2
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64, 
                out_channels=96, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 畳み込みブロック3
        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=96, 
                out_channels=128, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Global Average Pooling 層
        # これがシーケンス長に関係なく特徴量を集約するキーパーツ
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, d_output) # 入力はconv2の出力チャネル数である128
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global Average Poolingを適用
        x = self.global_avg_pool(x)
        
        # 分類器を適用
        x = self.classifier(x)
        
        return x, None