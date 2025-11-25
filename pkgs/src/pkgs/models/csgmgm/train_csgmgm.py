"""
2025-09-29

Define the training function of the CS-GMGM.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pkgs.models.csgmgm import layers_csgmgm, networks_csgmgm


def train_logistic(
    dataset_file: Path,
    out_dir: Path,
    label_list: List[str],
    z_dim: int,
    w_dim: int,
    wave_len: int,
    num_epoch: int,
    batch_size: int,
    device: torch.device,
    g_train_num: int = 4,
    d_train_num: int = 4,
    g_reg_int: int = 4,
    d_reg_int: int = 16,
    lr: float = 0.002,
    beta1: float = 0.0,
    beta2: float = 0.99,
    save_iter: int = 100,
    save_iter_detail: int = 5000,
) -> None:
    # Setting hyper-parameters of the Adam
    g_lr, g_beta1, g_beta2 = layers_csgmgm.adjust_adam_params(g_reg_int, lr, beta1, beta2)
    d_lr, d_beta1, d_beta2 = layers_csgmgm.adjust_adam_params(d_reg_int, lr, beta1, beta2)

    # Set the seed value for the random number generator
    torch.manual_seed(1234)

    assert dataset_file.exists(), "Training dataset csv file not found"

    # Save directory
    assert out_dir.exists(), "Output directory not found"

    # Initialize the DNN
    label_dim = len(label_list)
    gen_label_dim = label_dim - 1
    netG = networks_csgmgm.Generator(z_dim=z_dim, c_dim=gen_label_dim, w_dim=w_dim, wave_len=wave_len).to(device)
    netD = networks_csgmgm.Discriminator(label_dim=label_dim).to(device)

    # Initialized for weight decay.
    # Note: This model is not used in our study.
    netG_pred = networks_csgmgm.Generator(z_dim=z_dim, c_dim=gen_label_dim, w_dim=w_dim, wave_len=wave_len).to(device)

    # Weight decay value
    decay = 0.5 ** (batch_size / (10 * 1000))

    # Optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(g_beta1, g_beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(d_beta1, d_beta2))

    # load the data
    wave_dataset = networks_csgmgm.GroundMotionDatasets(csv_path=dataset_file, label_list=label_list)
    data_loader = DataLoader(wave_dataset, batch_size=batch_size, shuffle=True)

    netG.train()
    netG_pred.eval()
    netD.train()

    # Weight decay
    networks_csgmgm.Generator.apply_decay_parameter(netG, netG_pred, decay_rate=0.0)

    g_loss_list = []
    g_loss_reg_list = []
    g_train_prob_list = []
    d_train_prob_list_real = []
    d_train_prob_list_fake = []
    d_loss_list = []
    d_loss_reg_list = []

    time = np.linspace(0.0, (wave_len - 1) * 0.01, wave_len)

    # Noise for checking the waveform of the generated ground motion.
    fixed_noise = torch.randn(30, z_dim).to(device)
    fixed_label = torch.Tensor([6.8, np.log(20), np.log10(300)]).repeat(30, 1).to(device)

    # Save data frame header
    col_names_1 = ["Epoch", "g_loss", "d_loss", "g_reg_loss", "d_reg_loss", "g_prob", "d_prob_real", "d_prob_fake"]

    loss_func_g_reg = layers_csgmgm.GeneratorLossPathRegularization(device=device, g_reg_int=g_reg_int).to(device)

    for epoch in range(num_epoch):
        # Train the discriminator
        layers_csgmgm.set_model_requires_grad(netG, flag=False)
        layers_csgmgm.set_model_requires_grad(netD, flag=True)

        temp_d_loss = []
        temp_d_train_prob_fake = []
        temp_d_train_prob_real = []
        temp_d_loss_reg = []

        data_itr = iter(data_loader)

        for ind in range(d_train_num):
            real_wave, real_label = next(data_itr)

            real_wave = real_wave.to(device)
            real_label = real_label.to(device)
            real_label_1 = real_label[:, :-1]

            train_z = torch.randn(real_wave.shape[0], z_dim).to(device)
            fake_wave, fake_style, fake_pga = netG(train_z, real_label_1, is_train=True)

            fake_label = torch.cat([real_label_1, fake_pga], dim=1)

            real_d_out = netD(real_wave, real_label)
            fake_d_out = netD(fake_wave.detach(), fake_label.detach())

            d_loss = layers_csgmgm.discriminator_logistic_loss(disc_fake_out=fake_d_out, disc_real_out=real_d_out)

            netD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            temp_d_loss.append(d_loss.item())
            temp_d_train_prob_fake.append(torch.sigmoid(fake_d_out).mean().item())
            temp_d_train_prob_real.append(torch.sigmoid(real_d_out).mean().item())

        d_loss_list.append(np.mean(temp_d_loss))
        d_train_prob_list_real.append(np.mean(temp_d_train_prob_real))
        d_train_prob_list_fake.append(np.mean(temp_d_train_prob_fake))

        # Normalization
        if epoch % d_reg_int == 0:
            # Train the same number of times as the discriminator logistic loss
            for _ in range(d_train_num):
                real_wave, real_label = next(data_itr)
                real_wave = real_wave.to(device)
                real_label = real_label.to(device)

                real_wave.requires_grad = True
                real_d_out = netD(real_wave, real_label)
                d_reg = layers_csgmgm.discriminator_loss_r1(disc_real_out=real_d_out, reals_f=real_wave)

                netD.zero_grad()
                d_reg.backward()
                optimizerD.step()

                temp_d_loss_reg.append(d_reg.item())

            d_loss_reg_list.append(np.mean(temp_d_loss_reg))
        else:
            d_loss_reg_list.append(0)

        # Train the generator
        layers_csgmgm.set_model_requires_grad(netG, flag=True)
        layers_csgmgm.set_model_requires_grad(netD, flag=False)

        temp_g_loss = []
        temp_g_loss_reg = []
        temp_g_train_prob_fake = []

        for ind in range(g_train_num):
            _, real_label = next(data_itr)
            real_label = real_label.to(device)
            real_label_1 = real_label[:, :-1]

            train_z = torch.randn(size=(batch_size, z_dim)).to(device)
            fake_wave, fake_style, fake_pga = netG(train_z, real_label_1, is_train=True)
            fake_label = torch.cat([real_label_1, fake_pga], dim=1)
            fake_d_out = netD(fake_wave, fake_label)

            if epoch == 0 and ind == 0:
                print("gen wave size: {}".format(fake_wave.shape))
                print("disc out size: {}".format(fake_d_out.shape))

            g_loss = layers_csgmgm.generator_logistic_loss(fake_d_out)

            netG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            temp_g_loss.append(g_loss.item())
            temp_g_train_prob_fake.append(torch.sigmoid(fake_d_out).mean().item())

        g_loss_list.append(np.mean(temp_g_loss))
        g_train_prob_list.append(np.mean(temp_g_train_prob_fake))

        # Normalization
        if epoch % g_reg_int == 0:
            # Train the same number of times as the generator logistic loss
            for _ in range(g_train_num):
                _, real_label = next(data_itr)
                real_label = real_label.to(device)
                real_label_1 = real_label[:, :-1]

                train_z = torch.randn(size=(batch_size, z_dim)).to(device)
                fake_wave, fake_style, _ = netG(train_z, real_label_1, is_train=True)
                gen_reg, _ = loss_func_g_reg(fake_wave=fake_wave, fake_style=fake_style)
                netG.zero_grad()
                gen_reg.backward()
                optimizerG.step()

                temp_g_loss_reg.append(gen_reg.item())

            g_loss_reg_list.append(np.mean(temp_g_loss_reg))
        else:
            g_loss_reg_list.append(0)

        # Weight decay
        networks_csgmgm.Generator.apply_decay_parameter(netG, netG_pred, decay_rate=decay)

        # Save the generated data figure
        if epoch % save_iter == 0:
            netG.eval()

            fake_wave_valid, _, fake_pga_valid = netG(fixed_noise, fixed_label, is_train=False)
            out_data = fake_wave_valid.to("cpu").detach().numpy().squeeze()
            out_label = fake_pga_valid.to("cpu").detach().numpy().squeeze()

            fig, axes = plt.subplots(
                10, 3, figsize=(18, 12), sharex="all", sharey="all", subplot_kw=dict(xlim=(0.0, wave_len * 0.01))
            )
            axs = plt.gcf().get_axes()

            for ii, ax in enumerate(axs):
                ax.plot(time, out_data[ii, :], lw=0.2, c="slateblue")
                ax.text(0.02, 0.95, out_label[ii], ha="left", va="top", transform=ax.transAxes, size=8)

            fig.savefig(out_dir / f"gen_wave_cs_epoch_{epoch + 1}.png")
            plt.clf()
            plt.close()

            netG.train()

        print("[Epoch: {}/{}] [D loss: {}] [G loss: {}]".format(epoch, num_epoch, g_loss_list[-1], d_loss_list[-1]))

        if (epoch + 1) == 30000:
            save_iter = save_iter_detail

        if (epoch + 1) % save_iter == 0:
            # Save the model
            temp_state_dict_G = netG.state_dict()
            torch.save(temp_state_dict_G, out_dir / f"model_G_cs_epoch_{epoch + 1}.pth")

            temp_state_dict_D = netD.state_dict()
            torch.save(temp_state_dict_D, out_dir / f"model_D_cs_epoch_{epoch + 1}.pth")

            # Save the loss
            temp_out = np.stack(
                [
                    np.arange(epoch + 1),
                    np.array(g_loss_list),
                    np.array(d_loss_list),
                    np.array(g_loss_reg_list),
                    np.array(d_loss_reg_list),
                    np.array(g_train_prob_list),
                    np.array(d_train_prob_list_real),
                    np.array(d_train_prob_list_fake),
                ],
                axis=1,
            )
            df1 = pd.DataFrame(temp_out, columns=col_names_1)
            df1.to_csv(out_dir / "results_all.csv", index=False)

    g_loss_list = np.array(g_loss_list)
    g_loss_reg_list = np.array(g_loss_reg_list)
    g_train_prob_list = np.array(g_train_prob_list)
    d_train_prob_list_real = np.array(d_train_prob_list_real)
    d_train_prob_list_fake = np.array(d_train_prob_list_fake)
    d_loss_list = np.array(d_loss_list)
    d_loss_reg_list = np.array(d_loss_reg_list)

    out_mat_1 = np.stack(
        [
            np.arange(num_epoch),
            g_loss_list,
            d_loss_list,
            g_loss_reg_list,
            d_loss_reg_list,
            g_train_prob_list,
            d_train_prob_list_real,
            d_train_prob_list_fake,
        ],
        axis=1,
    )
    df1 = pd.DataFrame(out_mat_1, columns=col_names_1)
    df1.to_csv(out_dir / "results_all.csv", index=False)
