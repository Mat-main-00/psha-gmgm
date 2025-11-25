"""
2025-09-29

Training the S-GMGM.
"""

from pathlib import Path
import torch
from pkgs.models.sgmgm import train_sgmgm

dataset_file = Path("../data/dataset_sgmgm.csv")
out_dir = Path("../out_data/sgmgm")

# Header of the used condition labels
label_list = ["mw", "log_fault_dist", "log10_pga", "log10_v30"]

# Dimension of the latent variables
z_dim = 512
w_dim = 512

# Length of the target ground motion
wave_len = 8192

num_epoch = 100000
batch_size = 64

g_train_num = 4
d_train_num = 4
g_reg_int = 4
d_reg_int = 16

base_lr = 0.002
base_beta1 = 0.0
base_beta2 = 0.99

save_iter = 100
save_iter_detail = 100

# -----------------------------------------------------------------------------------------------
label_dim = len(label_list)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sgmgm.train_logistic(
        dataset_file=dataset_file,
        out_dir=out_dir,
        label_list=label_list,
        z_dim=z_dim,
        w_dim=w_dim,
        wave_len=wave_len,
        num_epoch=num_epoch,
        batch_size=batch_size,
        device=device,
        g_train_num=g_train_num,
        d_train_num=d_train_num,
        g_reg_int=g_reg_int,
        d_reg_int=d_reg_int,
        lr=base_lr,
        beta1=base_beta1,
        beta2=base_beta2,
        save_iter=save_iter,
        save_iter_detail=save_iter_detail,
    )
