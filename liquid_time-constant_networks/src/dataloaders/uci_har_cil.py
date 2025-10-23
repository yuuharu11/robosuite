import os
import numpy as np
import torch
from .base import SequenceDataset
from src.dataloaders.base import default_data_path

class UCIHAR_CIL(SequenceDataset):
    _name_ = "uci_har_cil"
    d_input = 9
    d_output = 6
    l_output = 0 #絶対に必要 ないとdecoderでエラーが出る
    L = 128

    def __init__(self, data_dir=None, val_split=0.2, seed=42, task_id=0, d_output=6, overall=False, **kwargs):
        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed
        self.task_id = task_id
        self.d_output = d_output
        self.overall = overall

        # --- CILシナリオの定義 ---
        # 各タスクで学習する「新しい」クラスのリスト
        self.tasks = [
            [0,1],
            [2,3],
            [4,5],
        ]
        # UCI-HAR Labels: 0-WALKING, 1-WALKING_UPSTAIRS, 2-WALKING_DOWNSTAIRS, 3-SITTING, 4-STANDING, 5-LAYING

        # これまでの全タスクのクラスを結合
        self.visible_classes = []
        if overall:
            for i in range(task_id + 1):
                self.visible_classes.extend(self.tasks[i])
        else:
            self.visible_classes = self.tasks[task_id]
        # setup呼び出し
        self.setup()

    @property
    def init_defaults(self):
        return {
            "val_split": 0.2,
            "seed": 42,
            "normalize": True,
            "task_id": 0
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "uci_har"
        data_path = self.data_dir / "UCI HAR Dataset"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found in {self.data_dir}.")

        # --- ステップ1: データ読み込み ---
        X_train_full = self._load_X(data_path / "train/Inertial Signals/", "train")
        y_train_full = self._load_y(data_path / "train/y_train.txt")
        X_test_full = self._load_X(data_path / "test/Inertial Signals/", "test")
        y_test_full = self._load_y(data_path / "test/y_test.txt")

        # --- ステップ2: フィルタリング ---
        print(f"--- CIL Task {self.task_id} : Filtering for classes {self.visible_classes} ---")

        train_mask = np.isin(y_train_full, self.visible_classes)
        X_train_visible, y_train_visible = X_train_full[train_mask], y_train_full[train_mask]

        test_mask = np.isin(y_test_full, self.visible_classes)
        X_test_visible, y_test_visible = X_test_full[test_mask], y_test_full[test_mask]

        # --- ステップ3: train/val分割 ---
        np.random.seed(self.seed)
        n_train_samples = len(X_train_visible)
        indices = np.random.permutation(n_train_samples)
        val_size = int(n_train_samples * self.val_split)
        val_indices = indices[-val_size:]
        train_indices = indices[:-val_size]

        X_train_split, y_train_split = X_train_visible[train_indices], y_train_visible[train_indices]
        X_val_split, y_val_split = X_train_visible[val_indices], y_train_visible[val_indices]

        # 正規化
        if getattr(self, "normalize", True):
            mean = np.mean(X_train_split, axis=(0, 1), keepdims=True)
            std = np.std(X_train_split, axis=(0, 1), keepdims=True)
            X_train_split = (X_train_split - mean) / (std + 1e-8)
            X_val_split = (X_val_split - mean) / (std + 1e-8)
            X_test_visible = (X_test_visible - mean) / (std + 1e-8)

        # TensorDataset化
        self.dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train_split).float(), torch.from_numpy(y_train_split).long())
        self.dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test_visible).float(), torch.from_numpy(y_test_visible).long())
        self.dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(X_val_split).float(), torch.from_numpy(y_val_split).long())

    def _to_tensor_dataset(self, x, y):
        x_tensor = torch.from_numpy(x).float().permute(0, 2, 1)
        y_tensor = torch.from_numpy(y).long()
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)

    def _load_X(self, path, split="train"):
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"
        ]
        X = [np.loadtxt(path / f"{sig}_{split}.txt", dtype=np.float32) for sig in signals]
        return np.transpose(np.array(X), (1, 2, 0))

    def _load_y(self, path):
        return np.loadtxt(path, dtype=np.int32) - 1
