import h5py
import numpy as np

# HDF5 ファイルのパス
hdf5_path = "/work/robomimic/datasets/lift/ph/low_dim_v15.hdf5"

with h5py.File(hdf5_path, "r") as f:
    # データ構造を確認
    print("Top-level keys:", list(f.keys()))
    
    # robomimicのHDF5構造: data/demo_X/{obs, actions, ...}
    data_group = f["data"]
    demo_keys = list(data_group.keys())
    print(f"Number of demonstrations: {len(demo_keys)}")
    
    # 最初のデモから時間軸の長さを取得
    first_demo = data_group[demo_keys[0]]
    print(f"First demo keys: {list(first_demo.keys())}")
    
    # actionsの長さからタイムステップ数を推定
    actions = first_demo["actions"][:]
    data_len = len(actions)
    print(f"Number of timesteps in first demo: {data_len}")
    
    # robomimic/robosuiteのデフォルトは通常20Hz (dt=0.05s)
    dt = 0.05  # s
    hz = 1.0 / dt
    print(f"Estimated sampling frequency: {hz} Hz")
    print(f"Total duration of first demo: {data_len * dt:.2f} seconds")
