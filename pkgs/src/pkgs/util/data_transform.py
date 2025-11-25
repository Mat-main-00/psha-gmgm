import numpy as np


def norm_data(target_data: np.ndarray, norm_data: np.ndarray) -> np.ndarray:
    """
    Normalize the conditional labels using the distribution of the observed records.
    The input is assumed to be a 1-dimensional numpy array, and the normalization is performed in the following three steps:
    1. Center the data by subtracting the mean.
    2. Scale it by the maximum absolute value.
    3. Adjust the standard deviation to 0.1.
    """
    if target_data.ndim != 1:
        raise ValueError(f"target_data must be 1-dimensional, but got shape {target_data.shape}")
    if norm_data.ndim != 1:
        raise ValueError(f"norm_data must be 1-dimensional, but got shape {norm_data.shape}")

    coe_1 = np.mean(norm_data)
    temp = norm_data - coe_1
    coe_2 = np.max(np.abs(temp))
    temp = temp / coe_2
    coe_3 = np.std(temp) / 0.1

    x = target_data - coe_1
    x = x / coe_2
    x = x / coe_3

    return x


def denorm_data(target_data: np.ndarray, norm_data: np.ndarray) -> np.ndarray:
    """
    Restore the normalized data to its original scale.
    target_data: the normalized data
    norm_data: the data used for computing the normalization parameters
    """
    if target_data.ndim != 1:
        raise ValueError(f"target_data must be 1-dimensional, but got shape {target_data.shape}")
    if norm_data.ndim != 1:
        raise ValueError(f"norm_data must be 1-dimensional, but got shape {norm_data.shape}")

    coe_1 = np.mean(norm_data)
    temp = norm_data - coe_1
    coe_2 = np.max(np.abs(temp))
    temp = temp / coe_2
    coe_3 = np.std(temp) / 0.1

    denormed = target_data * coe_3
    denormed = denormed * coe_2
    denormed = denormed + coe_1

    return denormed
