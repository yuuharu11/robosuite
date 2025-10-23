import pandas as pd
import os

def extract_joint_training(input_path: str, output_path: str):
    """
    CILの実験ログCSVからJoint Trainingの結果のみを抽出し、新しいCSVファイルとして保存する。
    CSVの列数を自動検出し、ヘッダーが不完全な場合でも修正するロバストなバージョン。
    「各task_idにおける2回目のtraining」をJoint Trainingと判定する。

    Args:
        input_path (str): 入力となる元のCSVファイルのパス。
        output_path (str): 抽出結果を保存する新しいCSVファイルのパス。
    """
    # --- 1. CSVの読み込みとヘッダーの自動修正 ---
    try:
        # まず最初のデータ行を読み込み、列数を確認する
        with open(input_path, 'r') as f:
            # ヘッダー行をスキップして最初のデータ行を読む
            header = f.readline()
            first_line = f.readline()
            num_columns = len(first_line.split(','))
        
        # 正しいヘッダーの完全なリスト
        correct_headers_17 = [
            'phase', 'phase_type', 'train_seed', 'test_seed', 'noise_level', 'task_id', 
            'テスト精度 (Test Acc)', '平均レイテンシ (ms/バッチ)', 'p95 レイテンシ (ms/バッチ)', 
            '推論時 Memory Allocated [MB]', '推論時 Memory Reserved [MB]', '学習時間/epoch', 
            '訓練時 Memory Allocated [MB]', '訓練時 Memory Reserved [MB]', '検証精度 (Val Acc)', 
            'チェックポイントパス', 'wandbリンク'
        ]
        correct_headers_16 = [h for h in correct_headers_17 if h != 'task_id']

        # ヘッダーに'task_id'がなく、かつデータの列数が17の場合 -> ヘッダーが不完全と判断
        if 'task_id' not in header and num_columns == 17:
            print("情報: ヘッダーは不完全ですがデータは17列です。ヘッダーを修正して読み込みます。")
            df = pd.read_csv(input_path, header=0, names=correct_headers_17)
        # ヘッダーに'task_id'がなく、かつデータの列数が16の場合
        elif 'task_id' not in header and num_columns == 16:
            print("情報: 'task_id'カラムが欠落しています。ヘッダーを修正して読み込みます。")
            df = pd.read_csv(input_path, header=0, names=correct_headers_16)
        else:
            # ヘッダーが正常な場合
            df = pd.read_csv(input_path)
            
        print(f"'{input_path}' を読み込みました。合計 {len(df)} 行。")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: '{input_path}'")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に予期せぬ問題が発生しました: {e}")
        return

    # --- 2. task_idの復元とJoint Trainingの抽出 ---
    if 'task_id' not in df.columns:
        df['task_id'] = df['チェックポイントパス'].str.extract(r'Task_(\d+)', expand=False)
        df.dropna(subset=['task_id'], inplace=True)
        df['task_id'] = pd.to_numeric(df['task_id'], errors='coerce').astype('Int64')

    training_df = df[df['phase_type'] == 'training'].copy()

    if training_df.empty:
        print("警告: 'phase_type'が'training'の行が見つかりませんでした。")
        return

    training_df['run_order'] = training_df.groupby('task_id').cumcount()
    joint_training_df = training_df[training_df['run_order'] == 1].copy()
    joint_training_df.drop(columns=['run_order'], inplace=True)

    if joint_training_df.empty:
        print("警告: Joint Trainingに該当する行が見つかりませんでした。")
        return

    print(f"Joint Trainingの結果を {len(joint_training_df)} 行抽出しました。")

    # --- 3. 新しいCSVとして出力 ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joint_training_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"抽出結果を '{output_path}' に保存しました。")


# --- スクリプトの実行 ---
if __name__ == "__main__":
    base_log_directory = "/work/csv/uci-har/cil/"
    output_joint_directory = "/work/csv/uci-har/cil-joint/"
    models = ["cnn.csv", "rnn.csv", "lstm.csv", "ltc_ncps.csv"]

    for model_file in models:
        input_csv_path = os.path.join(base_log_directory, model_file)
        output_csv_path = os.path.join(output_joint_directory, model_file)
        
        print(f"\n--- 処理開始: {model_file} ---")
        if os.path.exists(input_csv_path):
            extract_joint_training(input_csv_path, output_csv_path)
        else:
            print(f"ファイルが見つかりません: {input_csv_path}")