"""
2025-09-29

Define the training function of the CW-GMGM.
"""

from typing import List
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# load gan models
from pkgs.models.cwgmgm import networks_cwgmgm


def train_logistic(
    dataset_file: Path,
    out_dir: Path,
    label_list: List[str],
    z_dim: int,
    wave_len: int,
    num_epoch: int,
    batch_size: int,
    device: torch.device,
    lr: float,
    beta1: float = 0.0,
    beta2: float = 0.99,
    n_critic: int = 12,
    gp_lambda: float = 10.0,
    save_iter: int = 10,
) -> None:
    # Set the seed value for the random number generator
    torch.manual_seed(1234)

    assert dataset_file.exists(), "Training dataset csv file not found"

    # Save directory
    assert out_dir.exists(), "Output directory not found"

    # Initialize the DNNs
    netG = networks_cwgmgm.Generator(z_size=z_dim).to(device)
    netD = networks_cwgmgm.Discriminator(data_len=wave_len).to(device)

    # Optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    # Data loader
    wave_dataset = networks_cwgmgm.GroundMotionDatasets(csv_path=dataset_file, label_list=label_list)
    data_loader = DataLoader(wave_dataset, batch_size=batch_size, shuffle=True)

    # Lists to save the loss values
    log_G_losses = []
    log_D_losses = []
    log_epoch = []
    log_iteration = []
    log_i = []

    log_epoch_2 = []
    log_D_losses_2 = []
    log_G_losses_2 = []
    log_Wasserstein_D_2 = []

    iteration = 0
    batches_done = 0

    col_names_1 = ["Epoch", "g_loss", "d_loss", "g_prob", "d_prob_real", "d_prob_fake"]

    netG.train()
    netD.train()

    for epoch in range(num_epoch):
        temp_G_loss = []
        temp_D_loss = []
        temp_Wasserstein_D = []
        data_iter = iter(data_loader)
        total_batches = len(data_loader)
        groups = total_batches // n_critic

        temp_errG = 0.0
        d_iter_in_epoch = 0

        for _ in range(groups):
            # Update discriminator
            for _ in range(n_critic):
                # Get data
                real_wave, real_label = next(data_iter)
                real_wave = real_wave.to(device)
                real_label = real_label.to(device)
                sample_size = real_wave.size(0)

                # labels
                in_mw = real_label[:, 0:1].to(device)
                in_rrup = real_label[:, 1:2].to(device)
                in_vs30 = real_label[:, 2:3].to(device)
                in_pga = real_label[:, 3:4].to(device)

                # Input noise
                noise = torch.randn(sample_size, 1, z_dim).to(device)
                fake_wave, fake_label = netG(noise, in_mw, in_rrup, in_vs30)

                # Discriminator
                d_out_fake = netD(fake_wave.detach(), fake_label.detach(), in_mw, in_rrup, in_vs30)
                d_out_real = netD(real_wave, in_pga, in_mw, in_rrup, in_vs30)

                # Gradient penalty
                alpha = torch.rand(sample_size, 1, 1, 1, device=device)
                alpha_cn = alpha.view(sample_size, 1)

                xwf_hat = (alpha * real_wave + (1 - alpha) * fake_wave.detach()).requires_grad_(True)
                xcn_hat = (alpha_cn * in_pga + (1 - alpha_cn) * fake_label.detach()).requires_grad_(True)

                d_hat = netD(xwf_hat, xcn_hat, in_mw, in_rrup, in_vs30)
                grad_outputs = torch.ones_like(d_hat)

                # Grad of waveform
                grads_wf = torch.autograd.grad(
                    outputs=d_hat,
                    inputs=xwf_hat,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0].view(sample_size, -1)

                # Grad of PGA
                grads_cn = torch.autograd.grad(
                    outputs=d_hat,
                    inputs=xcn_hat,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0].view(sample_size, -1)

                # Gradient penalty
                grad_norm = torch.norm(torch.cat([grads_wf, grads_cn], dim=1), 2, dim=1)

                d_loss_gp = gp_lambda * (grad_norm - 1).square().mean()
                d_loss_w = -torch.mean(d_out_real) + torch.mean(d_out_fake)
                d_loss = d_loss_w + d_loss_gp

                netD.zero_grad()
                d_loss.backward()
                optimizerD.step()

                Wasserstein_D = -d_loss_w
                temp_D_loss.append(d_loss.item())
                temp_Wasserstein_D.append(Wasserstein_D.item())

                d_iter_in_epoch += 1
                iteration += 1
                batches_done += 1

                if batches_done % 5 == 0:
                    print(
                        f"[Epoch {epoch}/{num_epoch}] [Batch {batches_done % len(data_loader)}/{len(data_loader)}]"
                        f" [D loss: {d_loss.item():.4f}] [G loss: {temp_errG:.4f}]"
                    )

            # Generator
            sample_size = batch_size
            idxs = np.random.choice(len(wave_dataset), sample_size, replace=True)
            cond = torch.stack([wave_dataset[int(i)][1] for i in idxs]).to(device)
            rand_mw, rand_rrup, rand_vs30 = cond[:, 0:1], cond[:, 1:2], cond[:, 2:3]

            noise = torch.randn(sample_size, 1, z_dim).to(device)
            fake_wave, fake_label = netG(noise, rand_mw, rand_rrup, rand_vs30)
            errG = -torch.mean(netD(fake_wave, fake_label, rand_mw, rand_rrup, rand_vs30))

            netG.zero_grad()
            errG.backward()
            optimizerG.step()

            temp_errG = errG.item()
            temp_G_loss.append(temp_errG)

            log_epoch.append(epoch)
            log_i.append(d_iter_in_epoch - 1)
            log_iteration.append(iteration)
            log_D_losses.append(d_loss.item())
            log_G_losses.append(errG.item())

        log_epoch_2.append(epoch)
        log_D_losses_2.append(np.mean(temp_D_loss))
        log_G_losses_2.append(np.mean(temp_G_loss))
        log_Wasserstein_D_2.append(np.mean(temp_Wasserstein_D))

        # Save the model
        if (epoch + 1) % save_iter == 0:
            torch.save(netG.state_dict(), out_dir / f"model_G_cw_epoch_{epoch + 1}.pth")
            torch.save(netD.state_dict(), out_dir / f"model_D_cw_epoch_{epoch + 1}.pth")

    log_G_losses = np.array(log_G_losses)
    log_D_losses = np.array(log_D_losses)
    log_epoch = np.array(log_epoch)
    log_iteration = np.array(log_iteration)
    log_i = np.array(log_i)
    log_epoch_2 = np.array(log_epoch_2)
    log_D_losses_2 = np.array(log_D_losses_2)
    log_G_losses_2 = np.array(log_G_losses_2)
    log_Wasserstein_D_2 = np.array(log_Wasserstein_D_2)

    out_mat_1 = np.stack([log_epoch, log_i, log_iteration, log_D_losses, log_G_losses], axis=1)
    col_names_1 = ["Epoch", "index", "Iteration", "Loss_D", "Loss_G"]
    df1 = pd.DataFrame(out_mat_1, columns=pd.Index(col_names_1))
    df1.to_csv(out_dir / "results_all.csv", index=False)

    out_mat_2 = np.stack([log_epoch_2, log_D_losses_2, log_G_losses_2, log_Wasserstein_D_2], axis=1)
    col_names_2 = ["Epoch", "Loss_D: mean of each epochs", "Loss_G: the same as Loss_D", "Wasserstein_distance"]
    df2 = pd.DataFrame(out_mat_2, columns=pd.Index(col_names_2))
    df2.to_csv(out_dir / "results_mean.csv", index=False)
