import os
import zipfile
import requests
import numpy as np
import torch
from .base import SequenceDataset
from src.dataloaders.base import default_data_path

class UCIHAR(SequenceDataset):
    _name_ = "uci_har"
    d_input = 9
    d_output = 6
    l_output = 0
    L = 128

    @property
    def init_defaults(self):
        # normalize: True をデフォルトに追加
        return {"val_split": 0.2, 
                "seed": 42, 
                "normalize": True
            }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "uci_har"
        
        data_path = self.data_dir / "UCI HAR Dataset"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"UCI HAR Dataset not found in {self.data_dir}. "
                f"Please run the download commands in the terminal first."
            )

        X_train_path = data_path / "train/Inertial Signals/"
        y_train_path = data_path / "train/y_train.txt"
        X_test_path = data_path / "test/Inertial Signals/"
        y_test_path = data_path / "test/y_test.txt"

        X_train = self._load_X(X_train_path, "train")
        y_train = self._load_y(y_train_path)
        X_test = self._load_X(X_test_path, "test")
        y_test = self._load_y(y_test_path)

        # === ここから正規化処理を追加 ===
        if getattr(self, "normalize", True):
            print("Normalizing data...")
            # 1. 訓練データのみから平均と標準偏差を計算
            mean = np.mean(X_train, axis=(0, 1), keepdims=True)
            std = np.std(X_train, axis=(0, 1), keepdims=True)
            
            # 2. 訓練データとテストデータの両方に適用
            X_train = (X_train - mean) / (std + 1e-8)
            X_test = (X_test - mean) / (std + 1e-8)
        # ============================

        # 正規化されたデータを使ってTensorDatasetを作成
        self.dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        self.dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        
        # 訓練データをさらに学習用と検証用に分割
        self.split_train_val(self.val_split)

    def _load_X(self, path, split="train"):
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        X = []
        for sig_name in signals:
            file_name = f"{sig_name}_{split}.txt"
            file_path = os.path.join(path, file_name)
            signal = np.loadtxt(file_path, dtype=np.float32)
            X.append(signal)
        return np.transpose(np.array(X), (1, 2, 0))

    def _load_y(self, path):
        return np.loadtxt(path, dtype=np.int32) - 1