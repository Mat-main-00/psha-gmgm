"""
2025-08-13

PSHA sampling based on the CW-GMGM for numerical example 2.
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

# Source fault parameter settings
mw_list = [6.8, 6.8, 7.1, 6.9, 7.1, 6.6]
r_list = [20.1, 10.9, 19.2, 23, 14, 40.7]
model_list = ["bpt", "poisson", "bpt", "poisson", "bpt", "bpt"]
lam_list = [3250, 2500, 5650, 5000, 8000, 4000]
t_cur_list = [2755, 0, 1200, 0, 1955, 3100]
alp_list = [0.24, 0, 0.24, 0, 0.24, 0.24]
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
t_vs30_norm = data_transform.norm_data(np.log10(np.array([t_vs30])), obs_log10_v30_list)[0]

# Random number generator
rng = random.default_rng(seed)

event_list = []
save_wave_mat = []
save_int_mat = []
save_label_mat = []

sim_num = 0
total_event = 0

flag = True

time1 = time.time()

while flag:
    sim_num += 1
    for ff in range(len(mw_list)):
        num_event = 0

        # Probability model
        if model_list[ff] == "bpt":
            t_cur = t_cur_list[ff]
            mu = lam_list[ff]
            alpha = alp_list[ff]

            ft = ffm.bpt_cdf(t_cur, mu, alpha)
            ft_dt = ffm.bpt_cdf(t_cur + 50, mu, alpha)
            pt = (ft_dt - ft) / (1 - ft)
            u = rng.uniform(0, 1)
            if u <= pt:
                num_event = 1
            else:
                num_event = 0
        else:
            # Poisson model
            lam = lam_list[ff]

            # Number of event
            num_event = rng.poisson(lam=50 / lam, size=1)[0]

        # Record the number of occurred earthquakes
        if num_event == 0:
            continue
        else:
            total_event += num_event
            for i in range(num_event):
                event_list.append(ff)

            # Sampling ground motion data
            target_mw = np.repeat(mw_list[ff], num_event)
            target_r = np.repeat(r_list[ff], num_event)

            # De-normalize
            t_mw_norm = data_transform.norm_data(target_mw, obs_mw_list).reshape(-1, 1)
            t_r_norm = data_transform.norm_data(np.log(target_r), obs_logr_list).reshape(-1, 1)
            label_mat = np.concatenate(
                [t_mw_norm, t_r_norm, np.repeat(t_vs30_norm, num_event).reshape(-1, 1)], axis=1
            ).astype(np.float32)
            label_mat_t = torch.from_numpy(label_mat).to(device)
            noise = torch.randn(size=(num_event, 1, z_dim), device=device)

            # Generate data
            with torch.no_grad():
                fake_wave, fake_pga = netG(noise, label_mat_t[:, 0:1], label_mat_t[:, 1:2], label_mat_t[:, 2:3])
            save_acc = fake_wave.squeeze().to("cpu").detach().numpy()
            save_pga = fake_pga.squeeze().to("cpu").detach().numpy()

            # De-normalize PGA
            save_pga = 10 ** data_transform.denorm_data(np.atleast_1d(save_pga), obs_log10_pga_list)
            tmp = np.concatenate(
                [
                    np.repeat(sim_num, num_event).reshape(-1, 1),
                    np.repeat(ff, num_event).reshape(-1, 1),
                    target_mw.reshape(-1, 1),
                    target_r.reshape(-1, 1),
                    np.repeat(t_vs30, num_event).reshape(-1, 1),
                    save_pga.reshape(-1, 1),
                ],
                axis=1,
            )

            save_wave_mat.append(save_acc.reshape(-1, wave_len))
            save_label_mat.append(tmp)

    if len(save_wave_mat) == 0:
        continue

    if total_event >= finish_num:
        flag = False

# Save all sampling results
save_wave_tmp = np.concatenate(save_wave_mat, axis=0)
save_label_tmp = np.concatenate(save_label_mat, axis=0)

save_df = pd.DataFrame(save_label_tmp, columns=["sim_num", "event_num", "mw", "fault_dist", "v30", "gen_pga"])
if len(save_df) != len(event_list):
    raise ValueError("Multiple earthquakes are occurred")
save_df.to_csv(save_dir / f"sample_label_ex2_epoch_{epoch}.csv", index=False)
np.save(save_dir / f"sample_wave_ex2_epoch_{epoch}.npy", save_wave_tmp)

time2 = time.time()
elapsed = time2 - time1
print(f"Time: {elapsed}")

# Save in log file
with open("time_log_cw.txt", "a") as f:
    f.write(f"Elapsed time: {elapsed:.6f} seconds\n")
