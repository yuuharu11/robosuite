import os
import numpy as np
import torch
from .base import SequenceDataset
from src.dataloaders.base import default_data_path

class UCIHAR_DIL(SequenceDataset):
    _name_ = "uci_har_dil"
    d_input = 9
    d_output = 6
    l_output = 0
    L = 128

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # センサーグループの定義を一度だけ行う
        self.sensor_channels = {
            0: [6, 7, 8], # total_acc
            1: [0, 1, 2], # body_acc
            2: [3, 4, 5],  # body_gyro
            3: [0, 1, 2, 3, 4, 5], # body_acc + body_gyro
            4: [0, 1, 2, 6, 7, 8], # body_acc + total_acc
            5: [3, 4, 5, 6, 7, 8], # body_gyro + total_acc
            6: list(range(9)),      # all sensors
        }

    @property
    def init_defaults(self):
        return {
            "val_split": 0.2, 
            "seed": 42, 
            "normalize": True,
            "task_id": 0,
            "noise_level": 0.0,
            "joint_training": False # 合同学習
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "uci_har"
        
        data_path = self.data_dir / "UCI HAR Dataset"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"UCI HAR Dataset not found in {self.data_dir}. "
                f"Please download the dataset first."
            )

        X_train_clean = self._load_X(data_path / "train/Inertial Signals/", "train")
        y_train = self._load_y(data_path / "train/y_train.txt")
        X_test_clean = self._load_X(data_path / "test/Inertial Signals/", "test")
        y_test = self._load_y(data_path / "test/y_test.txt")

        if getattr(self, "normalize", True):
            mean = np.mean(X_train_clean, axis=(0, 1), keepdims=True)
            std = np.std(X_train_clean, axis=(0, 1), keepdims=True)
            X_train_clean = (X_train_clean - mean) / (std + 1e-8)
            X_test_clean = (X_test_clean - mean) / (std + 1e-8)
            
        if self.joint_training:
            print(f"--- Creating joint training dataset for tasks 0 to {self.task_id} ---")
            
            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
            
            # 現在のタスクIDまでの全タスクのデータを生成・結合する
            # (例: task_id=2 なら、タスク0, 1, 2 のデータセットを作成して結合)
            for i in range(self.task_id + 1):
                print(f"  - Generating data for task {i} with noise level {self.noise_level}...")
                
                # noise_level の値に応じて、ノイズ付加、無効化、またはクリーンなデータを適用
                if self.noise_level > 0:
                    # ガウシアンノイズを適用
                    data_std = np.std(X_train_clean)
                    noise_amplitude = data_std * self.noise_level
                    x_train_tmp = self._apply_noise(X_train_clean, i, noise_amplitude)
                    x_test_tmp = self._apply_noise(X_test_clean, i, noise_amplitude)
                elif self.noise_level < 0:
                    # センサー値を0にする無効化を適用
                    x_train_tmp = self._invalid_input(X_train_clean, i)
                    x_test_tmp = self._invalid_input(X_test_clean, i)
                else: # noise_level == 0
                    # クリーンなデータを使用
                    x_train_tmp = X_train_clean
                    x_test_tmp = X_test_clean

                X_train_list.append(x_train_tmp)
                X_test_list.append(x_test_tmp)
                y_train_list.append(y_train)
                y_test_list.append(y_test)

            # 全タスクのデータをまとめて連結
            X_train = np.concatenate(X_train_list, axis=0)
            X_test = np.concatenate(X_test_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
            
            print(f"--- Joint training dataset created. Total samples: {len(X_train)} ---")

        else:
            if self.noise_level > 0:
                print(f"Applying Gaussian noise (level: {self.noise_level}) to sensor group {self.task_id}...")
                np.random.seed(self.seed) # 再現性のためにseedを設定
                data_std = np.std(X_train_clean)
                noise_amplitude = data_std * self.noise_level
                X_train = self._apply_noise(X_train_clean, self.task_id, noise_amplitude)
                X_test = self._apply_noise(X_test_clean, self.task_id, noise_amplitude)
            elif self.noise_level < 0:
                print(f"Invalidating sensor group {self.task_id}...")
                X_train = self._invalid_input(X_train_clean, self.task_id)
                X_test = self._invalid_input(X_test_clean, self.task_id)
            else:
                X_train = X_train_clean
                X_test = X_test_clean

        self.dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        self.dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        
        self.split_train_val(self.val_split)

    def _apply_noise(self, data, task_id, noise_amplitude):
        noisy_data = np.copy(data)
        channels_to_corrupt = self.sensor_channels.get(task_id)
        if channels_to_corrupt is None:
            raise ValueError(f"Invalid task_id: {task_id}.")
        
        noise = np.random.normal(0, noise_amplitude, noisy_data[:, :, channels_to_corrupt].shape)
        noisy_data[:, :, channels_to_corrupt] += noise
        return noisy_data

    def _invalid_input(self, data, task_id):
        invalid_data = np.copy(data)
        channels_to_invalidate = self.sensor_channels.get(task_id)
        if channels_to_invalidate is None:
            raise ValueError(f"Invalid task_id: {task_id}.")
        
        invalid_data[:, :, channels_to_invalidate] = 0.0
        return invalid_data

    def _load_X(self, path, split="train"):
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        X = [np.loadtxt(path / f"{sig}_{split}.txt", dtype=np.float32) for sig in signals]
        return np.transpose(np.array(X), (1, 2, 0))

    def _load_y(self, path):
        return np.loadtxt(path, dtype=np.int32) - 1