# src/dataloaders/hapt.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from .base import SequenceDataset
from src.dataloaders.base import default_data_path
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

class HAPT(SequenceDataset):
    """
    HAPT (Human Activity and Postural Transitions) DataLoader
    - RawDataを読み込み、時系列ウィンドウを作成
    - 出力は (B, C, T) 形式
    """

    _name_ = "hapt"
    d_input = 6       # acc_x,y,z + gyro_x,y,z
    d_output = 12     # 12クラス
    window_len = 128  # 2.56秒 @50Hz

    @property
    def init_defaults(self):
        return {"val_split": 0.2, "seed": 42, "overlap": 0.5}

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "hapt"
        raw_dir = self.data_dir / "RawData"

        print("Loading HAPT raw data...")
        labels_path = raw_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"Cannot find {labels_path}")

        # ラベル読み込み
        labels_df = pd.read_csv(labels_path, sep=' ', header=None,
                                names=['exp_id', 'user_id', 'activity_id', 'start', 'end'])

        # Rawデータ読み込み
        all_df = self._load_all_raw_data(raw_dir, labels_df)

        # Train/Test分割
        train_subjects = pd.read_csv(self.data_dir / "Train/subject_id_train.txt", header=None).squeeze().unique()
        test_subjects  = pd.read_csv(self.data_dir / "Test/subject_id_test.txt", header=None).squeeze().unique()

        train_df = all_df[all_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
        test_df  = all_df[all_df['subject_id'].isin(test_subjects)].reset_index(drop=True)

        # 特徴量正規化
        feature_cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
        scaler = StandardScaler()
        train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df.loc[:, feature_cols]  = scaler.transform(test_df[feature_cols])

        # ウィンドウ化
        step = int(self.window_len * (1 - self.overlap))
        X_train, y_train = self._create_sequences(train_df, feature_cols, self.window_len, step)
        X_test, y_test   = self._create_sequences(test_df, feature_cols, self.window_len, step)

        """
        # (B, T, C) -> (B, C, T) に変換
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test  = np.transpose(X_test, (0, 2, 1))
        """
        # TensorDataset化
        self.dataset_train = TensorDataset(torch.from_numpy(X_train).float(),
                                           torch.from_numpy(y_train).long())
        self.dataset_test  = TensorDataset(torch.from_numpy(X_test).float(),
                                           torch.from_numpy(y_test).long())

        # 検証用分割
        self.split_train_val(self.val_split)

        print(f"Created {len(self.dataset_train)} training sequences and {len(self.dataset_test)} test sequences.")

    def _load_all_raw_data(self, raw_dir, labels_df):
        all_dfs = []
        unique_exps = labels_df[['exp_id','user_id']].drop_duplicates()

        for _, row in tqdm(unique_exps.iterrows(), total=len(unique_exps), desc="Loading Raw HAPT Files"):
            exp_id, user_id = row['exp_id'], row['user_id']
            acc_file = raw_dir / f'acc_exp{exp_id:02}_user{user_id:02}.txt'
            gyro_file = raw_dir / f'gyro_exp{exp_id:02}_user{user_id:02}.txt'

            df_acc = pd.read_csv(acc_file, sep=' ', header=None, names=['acc_x','acc_y','acc_z'])
            df_gyro = pd.read_csv(gyro_file, sep=' ', header=None, names=['gyro_x','gyro_y','gyro_z'])

            df_exp = pd.concat([df_acc, df_gyro], axis=1)
            df_exp['activity_id'] = 0
            df_exp['subject_id'] = user_id

            # ラベルを割り当て
            exp_labels = labels_df[labels_df['exp_id']==exp_id]
            for _, label_row in exp_labels.iterrows():
                start, end, act_id = label_row['start'], label_row['end'], label_row['activity_id']
                df_exp.iloc[start-1:end, df_exp.columns.get_loc('activity_id')] = act_id

            all_dfs.append(df_exp)

        full_df = pd.concat(all_dfs, ignore_index=True)
        # 0スタートに変換
        full_df['activity_id'] = full_df['activity_id'] - 1
        full_df = full_df[full_df['activity_id'] >= 0].reset_index(drop=True)
        return full_df

    def _create_sequences(self, df, feature_cols, window_len, step):
        sequences, labels = [], []
        for i in tqdm(range(0, len(df)-window_len, step), desc="Creating sequences"):
            window = df.iloc[i:i+window_len]
            label = int(window['activity_id'].mode()[0])  # ウィンドウの最頻値をラベルに
            sequences.append(window[feature_cols].values)
            labels.append(label)
        return np.array(sequences), np.array(labels)
