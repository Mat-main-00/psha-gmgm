"""
2025-09-29

Training the CW-GMGM.
"""

from pathlib import Path
import torch
from pkgs.models.cwgmgm import train_cwgmgm

torch.cuda.empty_cache()

dataset_file = Path("../data/dataset_cwgmgm.csv")
out_dir = Path("../out_data/cwgmgm")

# Header of the used condition labels
label_list = ["mw", "log_fault_dist", "log10_pga", "log10_v30"]

# Dimension of the latent variables
z_dim = 100

# Length of the target ground motion
wave_len = 4000

num_epoch = 1000
batch_size = 64

n_critic = 12

lr = 1e-4
beta1 = 0.0
beta2 = 0.99

gp_lambda = 10.0

save_iter = 10

# -----------------------------------------------------------------------------------------------
label_dim = len(label_list)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cwgmgm.train_logistic(
        dataset_file=dataset_file,
        out_dir=out_dir,
        label_list=label_list,
        z_dim=z_dim,
        wave_len=wave_len,
        num_epoch=num_epoch,
        batch_size=batch_size,
        device=device,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        n_critic=n_critic,
        gp_lambda=gp_lambda,
        save_iter=save_iter,
    )
