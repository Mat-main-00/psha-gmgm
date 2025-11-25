import numpy as np


def norm_data(target_data: np.ndarray, norm_data: np.ndarray) -> np.ndarray:
    """
    観測記録の分布を用いて条件ラベルを正規化する．
    cGANでの生成時に使用する．
    1次元のnumpy配列を前提とし、以下の3ステップで正規化を行う：
    1. 平均を引いて中心化
    2. 最大絶対値でスケーリング
    3. 標準偏差を0.1に調整
    """
    # 1次元配列かどうかをチェック
    if target_data.ndim != 1:
        raise ValueError(f"target_data must be 1-dimensional, but got shape {target_data.shape}")
    if norm_data.ndim != 1:
        raise ValueError(f"norm_data must be 1-dimensional, but got shape {norm_data.shape}")

    # norm_dataから正規化パラメータを計算（指定された形式）
    coe_1 = np.mean(norm_data)
    temp = norm_data - coe_1
    coe_2 = np.max(np.abs(temp))
    temp = temp / coe_2
    coe_3 = np.std(temp) / 0.1

    # Step 1: 平均を引いて中心化
    x = target_data - coe_1

    # Step 2: 最大絶対値でスケーリング
    x = x / coe_2

    # Step 3: 標準偏差を0.1に調整
    x = x / coe_3

    return x


def denorm_data(target_data: np.ndarray, norm_data: np.ndarray) -> np.ndarray:
    """
    正規化されたデータを元のスケールに戻す．
    target_data: 正規化されたデータ
    norm_data: 正規化するときに使用したデータ
    """
    # 1次元配列かどうかをチェック
    if target_data.ndim != 1:
        raise ValueError(f"target_data must be 1-dimensional, but got shape {target_data.shape}")
    if norm_data.ndim != 1:
        raise ValueError(f"norm_data must be 1-dimensional, but got shape {norm_data.shape}")

    # norm_dataから正規化パラメータを計算（指定された形式）
    coe_1 = np.mean(norm_data)
    temp = norm_data - coe_1
    coe_2 = np.max(np.abs(temp))
    temp = temp / coe_2
    coe_3 = np.std(temp) / 0.1

    # 逆変換を実行（norm_dataの逆順序）
    # Step 3の逆: 標準偏差を元に戻す
    denormed = target_data * coe_3

    # Step 2の逆: 最大絶対値でスケーリングを元に戻す
    denormed = denormed * coe_2

    # Step 1の逆: 平均を加える
    denormed = denormed + coe_1

    return denormed
