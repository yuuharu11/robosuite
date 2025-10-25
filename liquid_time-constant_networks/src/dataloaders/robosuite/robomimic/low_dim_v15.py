from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch
from src.dataloaders.base import SequenceDataset


def _to_str_list(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _read_split_demos(h5: h5py.File, split: str) -> List[str]:
    """
    HDF5のmask/{train,valid,test} からデモキーの配列を返す。
    なければ 9:1 の train:valid でフォールバック（testは空）。
    """
    if "mask" in h5:
        if split in h5["mask"]:
            return _to_str_list(h5["mask"][split][:])
        # "val" エイリアス
        if split == "val" and "valid" in h5["mask"]:
            return _to_str_list(h5["mask"]["valid"][:])
    # fallback
    all_demos = sorted(list(h5["data"].keys()))
    n = len(all_demos)
    if split == "train":
        return all_demos[: int(0.9 * n)]
    if split in ("val", "valid"):
        return all_demos[int(0.9 * n) :]
    if split == "test":
        return []
    return all_demos


class RobomimicLowDimV15(SequenceDataset):
    """
    robomimic low_dim_v15 (例: lift/ph) を読み込むSequenceDataset実装。
    - 観測は obs/{keys} を連結して (L, F)
    - 目的変数は actions を (L, A)（シーケンス回帰）
    - ウィンドウ長 L=seq_len でスライディング（stride）
    """

    _name_ = "robomimic_lowdim_v15"
    # d_input, d_output はセットアップ時に決定
    d_input = None
    d_output = None
    L = None
    l_output = 0

    def __init__(self, seed=42, val_split=0.2, seq_len=10, data_path=None, normalize=True, stride=1, obs_keys=None, **kwargs):
        self.seq_len = seq_len
        self.val_split = val_split
        self.seed = seed
        self.hdf5_path = data_path
        self.normalize = normalize
        self.stride = stride
        if obs_keys is None:
            obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        self.obs_keys = obs_keys

    @property
    def init_defaults(self):
        # UCIHARと同様に、ここでデフォルト引数を定義（設定ファイルから上書き）
        return {
            "hdf5_path": "/work/robomimic/datasets/lift/ph/low_dim_v15.hdf5",
            "obs_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            "stride": 1,
            "normalize": True,
            "seq_len": 10,
            "val_split": 0.2,
            "seed": 42,
        }

    def setup(self):
        hdf5_path = self.hdf5_path
        seq_len = int(self.seq_len)
        obs_keys: Sequence[str] = tuple(self.obs_keys)
        stride = int(self.stride)
        do_norm = bool(self.normalize)

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

        # まずtrain/valid/testのデモリストを取得
        with h5py.File(hdf5_path, "r") as f:
            train_demos = _read_split_demos(f, "train")
            # robomimicはtestが無いことが多いのでvalidをtestとして使う
            test_demos = _read_split_demos(f, "valid")
            data_grp = f["data"]

            # 1) trainウィンドウを構築
            X_train_list: List[np.ndarray] = []
            Y_train_list: List[np.ndarray] = []

            feat_dim = None
            act_dim = None

            for d in train_demos:
                if d not in data_grp:
                    continue
                g = data_grp[d]
                T = int(g["actions"].shape[0])
                if T < seq_len:
                    continue
                # 観測連結
                obs_arrs = [g["obs"][k][:] for k in obs_keys]  # list of (T, Dk)
                obs = np.concatenate(obs_arrs, axis=-1).astype(np.float32)  # (T, F)
                act = g["actions"][:].astype(np.float32)  # (T, A)

                if feat_dim is None:
                    feat_dim = int(obs.shape[-1])
                if act_dim is None:
                    act_dim = int(act.shape[-1])

                # スライディングウィンドウ
                for s in range(0, T - seq_len + 1, stride):
                    e = s + seq_len
                    X_train_list.append(obs[s:e])  # (L, F)
                    Y_train_list.append(act[s:e])  # (L, A)

            if len(X_train_list) == 0:
                raise RuntimeError("No training windows constructed. Check seq_len / stride / masks.")

            X_train = np.stack(X_train_list, axis=0)  # (N, L, F)
            Y_train = np.stack(Y_train_list, axis=0)  # (N, L, A)

            # 2) test(=valid)ウィンドウを構築
            X_test_list: List[np.ndarray] = []
            Y_test_list: List[np.ndarray] = []
            for d in test_demos:
                if d not in data_grp:
                    continue
                g = data_grp[d]
                T = int(g["actions"].shape[0])
                if T < seq_len:
                    continue
                obs_arrs = [g["obs"][k][:] for k in obs_keys]
                obs = np.concatenate(obs_arrs, axis=-1).astype(np.float32)
                act = g["actions"][:].astype(np.float32)
                for s in range(0, T - seq_len + 1, stride):
                    e = s + seq_len
                    X_test_list.append(obs[s:e])
                    Y_test_list.append(act[s:e])

            if len(X_test_list) > 0:
                X_test = np.stack(X_test_list, axis=0)
                Y_test = np.stack(Y_test_list, axis=0)
            else:
                # testが無い場合は空データセット
                X_test = np.zeros((0, seq_len, feat_dim), dtype=np.float32)
                Y_test = np.zeros((0, seq_len, act_dim), dtype=np.float32)

        # 正規化（trainの統計から）
        if do_norm:
            mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0, keepdims=True)  # (1, F)
            std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0, keepdims=True)   # (1, F)
            X_train = (X_train - mean) / (std + 1e-8)
            if X_test.shape[0] > 0:
                X_test = (X_test - mean) / (std + 1e-8)

        # TensorDataset へ
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()
        )

        # 学習用から検証用を分割
        self.split_train_val(self.val_split)

        # 形状メタを更新（モデルが参照）
        self.d_input = int(X_train.shape[-1])
        self.d_output = int(Y_train.shape[-1]) 
        self.L = int(seq_len)