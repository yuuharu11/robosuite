import torch

def add_salt_and_pepper_noise(tensor, noise_ratio, seed):
    """
    シード値に基づき、再現可能な「ごま塩ノイズ」を追加する。
    ピクセルをランダムに黒(0)または白(1)に置き換える。

    Args:
        tensor (torch.Tensor): 入力画像テンソル
        noise_ratio (float): ノイズを加えるピクセルの割合 (0.0 ~ 1.0)
        seed (int): 乱数生成のためのシード値

    Returns:
        torch.Tensor: ノイズが付与された画像テンソル
    """
    # 再現性のための独立した乱数ジェネレータ
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    
    # 元のテンソルをコピーして、直接変更しないようにする
    noisy_tensor = tensor.clone()
    
    # ノイズを加えるピクセル数を計算
    num_pixels = noisy_tensor.numel()
    num_noise_pixels = int(num_pixels * noise_ratio)
    
    # ノイズを加えるピクセルのインデックスをランダムに選ぶ
    noise_indices = torch.randperm(num_pixels, generator=generator, device=tensor.device)[:num_noise_pixels]
    
    # 半分を「塩」（白=1.0）、もう半分を「胡椒」（黒=0.0）にする
    num_salt = num_noise_pixels // 2
    salt_indices = noise_indices[:num_salt]
    pepper_indices = noise_indices[num_salt:]
    
    noisy_tensor.view(-1)[salt_indices] = 1.0
    noisy_tensor.view(-1)[pepper_indices] = 0.0
    
    return noisy_tensor
