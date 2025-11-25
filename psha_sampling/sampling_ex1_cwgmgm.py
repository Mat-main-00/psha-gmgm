"""
2025-07-04

PSHA sampling based on the CW-GMGM for numerical example 1.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from numpy import random

import torch
import pkgs.util.fault as ffm
from pkgs.util import data_transform
from pkgs.models.cwgmgm import networks_cwgmgm

torch.cuda.empty_cache()

# Parameters for sampling
lam = 0.5
t_range = 50
mu = 6.8
ml = 5.5
b = 0.9
area_r = 90
depth = 15
t_vs30 = 356

# Total number of loops for the Monte Carlo simulation
finish_num = 20000

# DNN parameter settings
z_dim = 100
wave_len = 4000
label_dim = 4
epoch = 250
col_name = ["mw", "log_fault_dist", "log10_pga", "log10_v30"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Model parameter file path
model_file = f"../data/model_CW_G_epoch_{epoch}.pth"

# Dataset label
label_file = f"../data/all_file_name_split_norm_cut_8192_rotation_45_data.csv"

# Random seed
seed = 1

# Output directory
save_dir = Path("../out_data/csgmgm/")
save_dir.mkdir(parents=True, exist_ok=True)

netG = networks_cwgmgm.Generator(z_size=z_dim).to(device)
state_dict = torch.load(model_file, map_location="cuda")
netG.load_state_dict(state_dict)
netG.eval()

# Observed records for data normalization.
obs_label_df = pd.read_csv(label_file)
obs_mw_list = obs_label_df["mw"].to_numpy()
obs_logr_list = obs_label_df["log_fault_dist"].to_numpy()
obs_log10_pga_list = obs_label_df["log10_pga"].to_numpy()
obs_log10_v30_list = obs_label_df["log10_v30"].to_numpy()

# Random number generator
rng = random.default_rng(seed)

# To save memory, split the results into multiple files.
div_num = 5
loop_num = int(finish_num / div_num)
t_vs30_norm = data_transform.norm_data(np.log10(np.array([t_vs30])), obs_log10_v30_list)[0]

time1 = time.time()


for jj in range(div_num):
    save_wave_mat = []
    save_label_mat = []

    for ll in range(loop_num):
        # Sample the event number and the corresponding m and r values
        num_event = rng.poisson(lam=lam * t_range, size=1)[0]
        uni = rng.uniform(0, 1, num_event)
        mw_list = ffm.sample_m(uni, b, mu, ml)
        temp_r = rng.uniform(0, 1, num_event)
        r_list = area_r * np.sqrt(temp_r)
        r_list = np.sqrt(r_list**2 + depth**2)

        # Normalize the conditional label values
        mw_list_norm = data_transform.norm_data(mw_list, obs_mw_list)
        r_list_norm = data_transform.norm_data(np.log(r_list), obs_logr_list)
        vs30_list_norm = np.repeat(t_vs30_norm, len(mw_list_norm))
        label_mat_t = torch.from_numpy(
            np.stack([mw_list_norm, r_list_norm, vs30_list_norm], axis=1).astype(np.float32)
        ).to(device)
        noise = torch.randn(size=(len(mw_list_norm), 1, z_dim), device=device)

        with torch.no_grad():
            fake_wave, fake_pga = netG(noise, label_mat_t[:, 0:1], label_mat_t[:, 1:2], label_mat_t[:, 2:3])
        save_acc = fake_wave.squeeze().to("cpu").detach().numpy()
        save_pga = fake_pga.squeeze().to("cpu").detach().numpy()
        save_pga = 10 ** data_transform.denorm_data(save_pga, obs_log10_pga_list)
        tmp = np.stack(
            [mw_list, r_list, np.repeat(t_vs30, len(mw_list)), save_pga, np.repeat(ll + 1, len(mw_list))], axis=1
        )

        save_wave_mat.append(save_acc)
        save_label_mat.append(tmp)

        if (ll + 1) % 100 == 0:
            print(f"Progress: [{ll + 1} / {loop_num}], [{jj} / {div_num}]")

    # Save results
    save_mat = np.concatenate(save_wave_mat, axis=0)
    save_label = np.concatenate(save_label_mat, axis=0)

    np.save(save_dir / f"sample_wave_poisson{jj + 1}_all_epoch_{epoch}.npy", save_mat)
    save_df = pd.DataFrame(save_label, columns=pd.Index(["mw", "fault_dist", "v30", "gen_pga", "sim_num"]))
    save_df.to_csv(save_dir / f"sample_label_poisson{jj + 1}_all_epoch_{epoch}.csv", index=False)

time2 = time.time()
elapsed = time2 - time1
print(f"Time: {elapsed}")

# Save in log file
with open("time_log_cw.txt", "a") as f:
    f.write(f"Elapsed time: {elapsed:.6f} seconds\n")
