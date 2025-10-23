# src/dataloaders/pamap2.py

import os
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from .base import SequenceDataset
from src.dataloaders.base import default_data_path
from sklearn.preprocessing import StandardScaler

class PAMAP2(SequenceDataset):
    """
    PAMAP2 (Physical Activity Monitoring) データセット用のデータローダークラス。
    
    ### データセットの主な特徴 ###
    - 被験者: 9人
    - センサー: 3つのIMU (手、胸、足首) と心拍数モニター
    - 活動クラス: 12種類 (歩行、サイクリング、アイロンがけ等) + 一時的な活動など (今回は12クラスに限定)
    - データ次元数: 54 (タイムスタンプや活動IDなどを含む) -> センサーデータは40次元を利用
    - 心拍1 + 各IMU(Temp1 + Acc6 + Acc16 + Gyro3 + Mag3 = 13)*3 = 40
    """
    _name_ = "pamap2"
    
    # --- データセットの基本情報を定義 ---
    d_input = 40      # 使用するセンサーの次元数 (IMU x 3 x 13次元 + 心拍数)
    d_output = 12     # 分類する活動クラスの数
    l_output = 0
    L = 171           # 論文等で一般的に使われるウィンドウサイズ (5.12秒 / 3) @100Hz 

    @property
    def init_defaults(self):
        """設定ファイルで指定されなかった場合のデフォルト値を定義"""
        return {"val_split": 0.2, "seed": 42, "window_size": 5.12, "overlap": 0.5}

    def setup(self):
        """
        データセットのセットアップを行うメインのメソッド。
        Hydraによってインスタンス化される際に呼び出される。
        """
        self.data_dir = self.data_dir or default_data_path / "pamap2"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = self.data_dir / "PAMAP2_Dataset"
        
        # --- 1. データのダウンロードと展開 ---
        if not dataset_path.exists():
            self._download_and_extract(dataset_path)

        # --- 2. データの読み込みと前処理 ---
        print("Loading and preprocessing PAMAP2 data...")
        all_data = self._load_data(dataset_path / 'Protocol')
        
        # --- 3. 被験者ごとに訓練/テストデータを分割 ---
        # データリークを防ぐため、特定の被験者をテスト用にする
        # subject 105, 106 をテストデータとするのが一般的
        train_subjects = [101, 102, 103, 104, 107, 108, 109]
        test_subjects = [105, 106]

        train_df = all_data[all_data['subject_id'].isin(train_subjects)]
        test_df = all_data[all_data['subject_id'].isin(test_subjects)]

        # --- 4. センサーデータの正規化 ---
        # 訓練データのみでStandardScalerを学習させ、両方に適用する
        scaler = StandardScaler()
        feature_columns = [col for col in train_df.columns if col not in ['timestamp', 'activity_id', 'heart_rate', 'subject_id']]
        
        train_df.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])
        test_df.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])

        # --- 5. スライディングウィンドウでシーケンスを作成 ---
        window_len = int(100 * self.window_size) # 100Hz * 5.12秒 = 512
        step = int(window_len * (1 - self.overlap))     # 512 * 0.5 = 256

        X_train, y_train = self._create_sequences(train_df, window_len, step)
        X_test, y_test = self._create_sequences(test_df, window_len, step)

        print(f"Created {len(X_train)} training sequences and {len(X_test)} test sequences.")
        
        # --- 6. PyTorchのTensorDatasetに変換 ---
        self.dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        self.dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        
        # --- 7. 訓練データから検証データを分割 ---
        self.split_train_val(self.val_split)

    def _download_and_extract(self, dataset_path):
        """データをWebからダウンロードし、zipファイルを展開する"""
        zip_path = self.data_dir / "PAMAP2_Dataset.zip"
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
        
        print("PAMAP2 dataset not found. Downloading...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print(f"Extracted to {dataset_path}")
            os.remove(zip_path)
        except requests.exceptions.RequestException as e:
            raise FileNotFoundError(f"Failed to download PAMAP2 dataset: {e}")

    def _load_data(self, data_path):
        """
        全被験者のデータを読み込み、前処理して一つのDataFrameに結合する。
        """
        # カラム名を定義
        columns = [
            'timestamp', 'activity_id', 'heart_rate',
            # IMU Hand
            'hand_temp', 'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z', 'hand_acc6_x', 'hand_acc6_y', 'hand_acc6_z',
            'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 'hand_mag_x', 'hand_mag_y', 'hand_mag_z', 'hand_orient_1', 'hand_orient_2', 'hand_orient_3', 'hand_orient_4',
            # IMU Chest
            'chest_temp', 'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z', 'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
            'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z', 'chest_orient_1', 'chest_orient_2', 'chest_orient_3', 'chest_orient_4',
            # IMU Ankle
            'ankle_temp', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z', 'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
            'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z', 'ankle_orient_1', 'ankle_orient_2', 'ankle_orient_3', 'ankle_orient_4'
        ]
        
        all_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
        df_list = []

        for filename in all_files:
            subject_id = int(filename.split('.')[0][-3:])
            df = pd.read_csv(os.path.join(data_path, filename), header=None, sep=' ', names=columns)
            df['subject_id'] = subject_id
            df_list.append(df)
            
        full_df = pd.concat(df_list, ignore_index=True)
        
        # --- データクレンジング ---
        # 1. 評価対象外の活動(ID=0)を除外
        full_df = full_df[full_df['activity_id'] != 0]

        # 2. ラベルを0から始まるように再マッピング (12クラス)
        # 8, 9, 10, 11, 12, 13, 16, 17, 24
        valid_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
        full_df = full_df[full_df['activity_id'].isin(valid_activities)]
        activity_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_activities)}
        full_df['activity_id'] = full_df['activity_id'].map(activity_mapping)
        
        # 3. 欠損値を線形補間で埋める
        full_df.interpolate(inplace=True)
        # それでも残る先頭のNaNは後方/前方で埋める
        full_df.fillna(method='bfill', inplace=True)
        full_df.fillna(method='ffill', inplace=True)

        # 4. Orientation (四元数) は扱いにくいため除外
        orientation_cols = [col for col in full_df.columns if 'orient' in col]
        full_df.drop(columns=orientation_cols, inplace=True)
        
        return full_df

    def _create_sequences(self, df, window_len, step):
        """
        DataFrameからスライディングウィンドウを用いてシーケンスデータを作成する。
        - 各ウィンドウのラベルは「最頻値 (mode)」を代表ラベルとする。
        - 全一致の制約を外すことで、遷移を含むウィンドウも利用可能にする。
        """
        sequences = []
        labels = []
        
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'activity_id', 'subject_id']]

        for i in range(0, len(df) - window_len, step):
            window = df.iloc[i: i + window_len]
            label = window['activity_id'].mode()[0]  # ウィンドウ内の最頻値をラベルにする

            sequences.append(window[feature_columns].values)
            labels.append(label)

        return np.array(sequences), np.array(labels)
