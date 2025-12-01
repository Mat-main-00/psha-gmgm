"""
2025-08-13

PSHA sampling based on the S-GMGM for numerical example 1.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy import random

import torch
import pkgs.util.fault as ffm
from pkgs.util import data_transform
from pkgs.models.sgmgm import networks_sgmgm

torch.cuda.empty_cache()

# Source fault parameter settings
mw_list = [6.8, 6.8, 7.1, 6.9, 7.1, 6.6]
r_list = [20.1, 10.9, 19.2, 23, 14, 40.7]
model_list = ["bpt", "poisson", "bpt", "poisson", "bpt", "bpt"]
lam_list = [3250, 2500, 5650, 5000, 8000, 4000]
t_cur_list = [2755, 0, 1200, 0, 1955, 3100]
alp_list = [0.24, 0, 0.24, 0, 0.24, 0.24]
t_vs30 = 356

# Tolerance for data acceptance
mw_diff = 0.05
r_diff = 5
vs30_diff = 5

# Total number of loops for the Monte Carlo simulation
finish_num = 20000

# Number of ground motions to generate at once
one_noise_num = 2048

# DNN parameter settings
epoch = 35300
z_dim = 512
w_dim = 512
col_name = ["mw", "log_fault_dist", "log10_pga", "log10_v30"]

# Model parameter file path
model_file = f"../data/model_G_epoch_{epoch}.pth"

# Dataset label
label_file = f"../data/all_file_name_split_norm_cut_8192_rotation_45_data.csv"

# Random seed
seed = 1

# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================

# Multi GPU settings
multi_flag = False

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        primary = torch.device(f"cuda:{device_ids[0]}")
        print(f"Using multiple GPUs: {device_ids}")
        print(f"Primary device: {primary}")
        multi_flag = True
    else:
        device = torch.device("cuda")
        print(f"Using single GPU: {device}")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Output directory
save_dir = Path("../out_data/sgmgm/")
save_dir.mkdir(parents=True, exist_ok=True)

# model
if multi_flag:
    netG = networks_sgmgm.Generator(z_dim=z_dim, w_dim=w_dim, label_dim=len(col_name), wave_len=8192).to(primary)
    netG.load_state_dict(torch.load(model_file, map_location=primary))

    # Modify
    syn = netG.synthesis_network  # alias
    syn.main_net_list = torch.nn.ModuleList(syn.main_net_list)
    syn.to_wave_list = torch.nn.ModuleList(syn.to_wave_list)

    netG = torch.nn.DataParallel(netG, device_ids=device_ids, output_device=device_ids[0])
else:
    netG = networks_sgmgm.Generator(z_dim=z_dim, w_dim=w_dim, label_dim=len(col_name), wave_len=8192).to(device)
    netG.load_state_dict(torch.load(model_file, map_location=device))

netG.eval()

# Load the values of dataset labels fir normalization
obs_label_df = pd.read_csv(label_file)
obs_mw_list = obs_label_df["mw"].to_numpy()
obs_logr_list = obs_label_df["log_fault_dist"].to_numpy()
obs_log10_pga_list = obs_label_df["log10_pga"].to_numpy()
obs_log10_v30_list = obs_label_df["log10_v30"].to_numpy()


# Random number generator
rng = random.default_rng(seed)

# Noise vector
if multi_flag:
    noise_buf = torch.empty(one_noise_num * len(device_ids), z_dim, device=primary)
else:
    noise_buf = torch.empty(one_noise_num, z_dim, device=device)

catalog = []
sim_num = 0

flag = True

time1 = time.time()

while flag:
    sim_num += 1
    for ff in range(len(mw_list)):
        num_event = 0

        # Fault model
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
            # Poisson
            lam = lam_list[ff]

            num_event = rng.poisson(lam=50 / lam, size=1)[0]

        if num_event == 0:
            continue
        else:
            for i in range(num_event):
                catalog.append([sim_num, ff, mw_list[ff], r_list[ff], t_vs30])

    if len(catalog) >= finish_num:
        flag = False

# Save catalog
catalog_df = pd.DataFrame(catalog, columns=["sim_num", "case_num", "mw", "r", "vs30"])

n_events = len(catalog_df)
print("Total events in catalog:", n_events)

print(catalog_df.head(10))

vs30_low = t_vs30 - vs30_diff
vs30_high = t_vs30 + vs30_diff
mw_target = catalog_df["mw"].to_numpy()
r_target = catalog_df["r"].to_numpy()
mw_low = mw_target - mw_diff
mw_high = mw_target + mw_diff
r_low = r_target - r_diff
r_high = r_target + r_diff

assigned = np.zeros(n_events, dtype=bool)

part_wave_list = [None] * n_events
part_label_list = [None] * n_events

while not np.all(assigned):
    noise_buf.normal_()
    with torch.no_grad():
        f_wave, _, f_label = netG(noise_buf, is_train=False)
    n_fake_label = f_label.squeeze().to("cpu").detach().numpy()

    # Denormalizing
    gen_mw = data_transform.denorm_data(n_fake_label[:, 0], obs_mw_list)
    gen_r = np.exp(data_transform.denorm_data(n_fake_label[:, 1], obs_logr_list))
    gen_vs30 = 10 ** data_transform.denorm_data(n_fake_label[:, 3], obs_log10_v30_list)

    # Vs30 mask
    vs_mask = (vs30_low <= gen_vs30) & (gen_vs30 <= vs30_high)
    cand_idx = np.where(vs_mask)[0]

    if len(cand_idx) == 0:
        continue

    save_acc = f_wave.squeeze().to("cpu").detach().numpy()
    gen_pga = 10 ** data_transform.denorm_data(n_fake_label[:, 2], obs_log10_pga_list)

    remaining = np.where(~assigned)[0]
    if len(remaining) == 0:
        break

    for ci in cand_idx:
        if len(remaining) == 0:
            break

        mw_c = gen_mw[ci]
        r_c = gen_r[ci]
        vs_c = gen_vs30[ci]

        mw_low_rem = mw_low[remaining]
        mw_high_rem = mw_high[remaining]
        r_low_rem = r_low[remaining]
        r_high_rem = r_high[remaining]

        mw_ok = (mw_low_rem <= mw_c) & (mw_c <= mw_high_rem)
        r_ok = (r_low_rem <= r_c) & (r_c <= r_high_rem)
        match_mask = mw_ok & r_ok

        if not np.any(match_mask):
            continue

        possible_events = remaining[match_mask]
        ev_idx = rng.choice(possible_events)

        # Save
        acc = save_acc[ci].squeeze()[None, :]  # shape (1, 8192)
        lab = np.array([mw_c, r_c, gen_pga[ci], vs_c], dtype=float).reshape(1, -1)

        part_wave_list[ev_idx] = acc
        part_label_list[ev_idx] = lab
        assigned[ev_idx] = True

        # 残りイベントを更新
        remaining = np.where(~assigned)[0]

    print("Rem: ", len(remaining))

# ===================== このブロックの保存 =====================
if not np.all(assigned):
    print(f"Warning: not fully assigned. Remaining: {np.sum(~assigned)}")
else:
    print(f"Completed.")

wave_arr = np.concatenate(part_wave_list, axis=0)
label_arr = np.concatenate(part_label_list, axis=0)

np.save(save_dir / f"sample_wave_sgmgm_ex2.npy", wave_arr)

label_df = pd.DataFrame(label_arr, columns=["mw_gen", "fault_dist_gen", "gen_pga", "v30_gen"])
label_df.to_csv(save_dir / f"sample_label_sgmgm_ex2.csv", index=False)

meta_df = catalog_df[["sim_num", "case_num", "mw", "r", "vs30"]].copy()
meta_df.to_csv(save_dir / f"sample_ind_sgmgm_ex2.csv", index=False)

time2 = time.time()
elapsed = time2 - time1
print(f"Time: {elapsed}")

# Save in log file
with open("time_log_ex2.txt", "a") as f:
    f.write(f"Elapsed time: {elapsed:.6f} seconds\n")
