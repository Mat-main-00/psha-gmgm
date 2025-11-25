"""
2025-07-30

Calculate the Sinkhorn divergence between the distributions of feature indices
of observed records (training dataset) and those of generated ground-motion data.
The Sinkhorn divergence is calculated for each epoch, and the analysis results are saved in a single CSV file with a shape of (N, 2).
N is the number of candidate models (i.e., models trained with different epochs);
the first column lists the epochs, and the second column represents the corresponding Sinkhorn divergence values.

Input:
    CSV files containing the feature index values of the observed records and those of the generated ground motions.
    "obs_features.csv" and "gen_features_epoch_*.csv" must be prepared in advance.

The trained model parameter files are not provided in this repository.
Running this script requires a GMGM .pth file that you have trained on your own dataset.

This script is an example for the CS-GMGM.
For the S-GMGM and CW-GMGM, the same script was applied for calculations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss

data_dir = Path("../../../../data/")

# Target epoch list
epoch_list = np.arange(10, 1010, 10)

# Target indices.
col_name = [
    "pga",
    "pgv",
    "pgv_pga",
    "arias",
    "si",
    "Tm",
    "D_5_95",
    "D_5_45",
    "zero_cross_num",
    "extreme_val",
    "predominant_freq",
    "seismic_intensity",
]

# Indices calculated in log-scale.
log_list = {"pga", "pgv", "pgv_pga", "arias", "si"}

device = torch.device("cuda")
dtype = torch.float32

# Observed record
obs_df = pd.read_csv(data_dir / "obs_features.csv")

# Mean and standard deviation
obs_mean_list = []
obs_std_list = []
obs_arr = []

for col in col_name:
    x = obs_df[col].to_numpy()

    if col in log_list:
        x = np.log10(x)

    x_m = np.mean(x)
    x_s = np.std(x)

    x = (x - x_m) / x_s

    obs_arr.append(x)
    obs_mean_list.append(x_m)
    obs_std_list.append(x_s)

# transform to torch.Tensor
x_obs = torch.from_numpy(np.stack(obs_arr, axis=1).astype(np.float32)).to(device)
print(x_obs.size())
n_obs = x_obs.shape[0]
a_w = torch.full((n_obs,), 1.0 / n_obs, device=device, dtype=dtype)  # uniform weight


def robust_blur_scale(X, sample=2048, frac=0.05):
    # X: torch (N,D) on device
    n = X.shape[0]
    idx = torch.randperm(n, device=device)[: min(sample, n)]
    Xi = X[idx]  # (m,D)

    with torch.no_grad():
        d2 = torch.cdist(Xi, Xi)  # (m,m)
        med = torch.median(d2[d2 > 0.0])
        if torch.isnan(med) or med <= 0:
            med = torch.tensor(1.0, device=device)
        return float(med.item() * frac)


blur = robust_blur_scale(x_obs, sample=2048, frac=0.05)


# Distance calculation
sinkhorn = SamplesLoss(
    "sinkhorn",
    p=1,
    blur=blur,
    backend="online",
    scaling=0.9,
    debias=True,
    verbose=False,
)

# Main loop
records = []

with torch.no_grad():
    for idx, epoch in enumerate(epoch_list, 1):
        gen_df = pd.read_csv(data_dir / f"gen_feature_epoch_{epoch}.csv")

        gen_arr = []

        for ii, col in enumerate(col_name):
            x = gen_df[col].to_numpy()

            if col in log_list:
                x = np.log10(x)

            x = (x - obs_mean_list[ii]) / obs_std_list[ii]

            gen_arr.append(x)

        x_gen = torch.from_numpy(np.stack(gen_arr, axis=1).astype(np.float32)).to(device)
        n_gen = x_gen.shape[0]
        b_w = torch.full((n_gen,), 1.0 / n_gen, device=device, dtype=dtype)

        # Sinkhorn‑ε distance
        w1 = sinkhorn(a_w, x_obs, b_w, x_gen).item()

        records.append([epoch, w1])
        print(f"[{idx:4d}/{len(epoch_list)}] epoch={epoch:5d}  W1(Sinkhorn div., blur={blur:.3g}) = {w1:.6f}")

out_df = pd.DataFrame(records, columns=["epoch", "W1_12D"])
out_df.to_csv(data_dir / "em_dist_12d_cs.csv", index=False)
