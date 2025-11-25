"""
2025-09-05

CW-GMGM (Conditional WassersteinGAN-GP-based Ground-Motion Generative Model)
Construct the neural networks of CW-GMGM.

This code is based on the following GitHub repository, modified to fit the problem settings of our study.
    https://github.com/mflorezto/AI_EQ_Ground_Motion.git
"""

from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

torch.cuda.empty_cache()


class GroundMotionDatasets(Dataset):
    """
    Dataset class for ground motion data and label data.
    """

    def __init__(self, csv_path: Path, label_list: List[str]) -> None:
        """
        Parameters
        ----------
        :param csv_path: The path to the csv file containing the path of ground motion data and the values of the corresponding
            label data.
        :param label_list: List of label data names.
        """
        super(GroundMotionDatasets, self).__init__()

        df = pd.read_csv(csv_path)
        self.data_path = df["file_name"].astype(str).tolist()
        labels = df[label_list].to_numpy().astype(np.float32)

        # Normalize the condition labels
        labels = labels - np.mean(labels, axis=0, keepdims=True)
        labels = labels / np.max(np.abs(labels), axis=0, keepdims=True)
        labels = labels / (np.std(labels, axis=0, keepdims=True) / 0.1)

        self.label = torch.from_numpy(labels).clone()

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, index):
        path = str(self.data_path[index])
        temp_mat = np.load(path, allow_pickle=True)
        out = torch.from_numpy(temp_mat.astype(np.float32)).clone()
        out = out.reshape(1, -1, 1)

        out_label = self.label[index, :]

        return out, out_label


def embed(in_chan: int, out_chan: int) -> nn.Module:
    """
    Creates embedding network with 4 fully connected layers.
    Progressively grows the number of output nodes.
    """
    layers = nn.Sequential(
        nn.Linear(in_chan, 32),
        torch.nn.ReLU(),
        nn.Linear(32, 64),
        torch.nn.ReLU(),
        nn.Linear(64, 256),
        torch.nn.ReLU(),
        nn.Linear(256, 512),
        torch.nn.ReLU(),
        nn.Linear(512, 1024),
        torch.nn.ReLU(),
        nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        nn.Linear(2048, out_chan),
        torch.nn.ReLU(),
    )

    return layers


class Discriminator(nn.Module):
    def __init__(self, data_len: int) -> None:
        """
        Parameters
        ----------
        :param data_len: Length of the ground-motion data.
        """
        super(Discriminator, self).__init__()

        self.data_len = data_len

        # Embedding network for condition labels.
        # It must be initialized with the number of labels being considered.
        self.embed1 = embed(1, data_len)
        self.embed2 = embed(1, data_len)
        self.embed3 = embed(1, data_len)

        # Embedding network for PGA
        self.nn_cnorm = embed(1, data_len)

        # Concatenate condition labels, PGA, and ground-motions.
        # Expected input data shape is (5, data_len, 1).
        self.conv1 = nn.Conv2d(
            5,
            16,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )

        self.conv1b = nn.Conv2d(
            16,
            16,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        self.conv2b = nn.Conv2d(
            32,
            32,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        self.conv3b = nn.Conv2d(
            64,
            64,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        self.conv4 = nn.Conv2d(
            64,
            128,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        self.conv4b = nn.Conv2d(
            128,
            128,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        self.conv5 = nn.Conv2d(
            128,
            256,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        self.conv5b = nn.Conv2d(
            256,
            256,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )

        # Output layers
        self.fc0 = nn.Linear(125, 110)
        self.fc1 = nn.Linear(110, 128)
        self.fc1b = nn.Linear(128, 100)
        self.fc2 = nn.Linear(256 * 100, 1)

    def forward(
        self,
        x: torch.Tensor,
        ln_cn: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------

        :param x             1-D ground-motion data.
        :param ln_cn         PGA.
        :param v1, v2, v3    Condition labels. The number of condition labels are fixed.
        """

        #  Apply embeddings for the condition labels
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)
        v3 = self.embed3(v3)

        # Embedding PGA
        ln_cn = self.nn_cnorm(ln_cn)

        # Reshape to concatenate with ground-motion data
        v1 = v1.view(-1, 1, self.data_len, 1)
        v2 = v2.view(-1, 1, self.data_len, 1)
        v3 = v3.view(-1, 1, self.data_len, 1)
        ln_cn = ln_cn.view(-1, 1, self.data_len, 1)

        x = torch.cat(
            [
                x,
                ln_cn,
                v1,
                v2,
                v3,
            ],
            dim=1,
        )

        # Main
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv1b(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2b(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3b(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4b(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv5(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv5b(x)
        x = F.leaky_relu(x, 0.2)

        x = torch.squeeze(x, dim=3)

        x = self.fc0(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc1b(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 256 * 100)

        out = self.fc2(x)

        return out


def FCNC(n_vs: int = 150, hidden_1: int = 256, hidden_2: int = 512) -> nn.Module:
    """
    Fully connected layers for generating PGA value of the ground-motion data.
    """
    layers = nn.Sequential(
        nn.Linear(n_vs, hidden_1),
        torch.nn.ReLU(),
        nn.Linear(hidden_1, hidden_2),
        torch.nn.ReLU(),
        nn.Linear(hidden_2, hidden_2),
        torch.nn.ReLU(),
        nn.Linear(hidden_2, hidden_1),
        torch.nn.ReLU(),
        nn.Linear(hidden_1, 1),
        torch.nn.Tanh(),
    )

    return layers


class Generator(nn.Module):
    def __init__(self, z_size: int) -> None:
        super(Generator, self).__init__()

        # Fully-connected layer for noise vector
        # Input: (1, z_size)
        self.fc00 = nn.Linear(z_size, 150, bias=False)
        self.batchnorm00 = nn.BatchNorm1d(1)

        # Embedding networks for condition labels
        self.embed1 = embed(1, 150)
        self.embed2 = embed(1, 150)
        self.embed3 = embed(1, 150)

        # Concatenate condition labels and noise vector.
        # Expected input data shape is (4, 150, 1).
        self.conv0 = nn.Conv2d(
            4,
            6,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm0 = nn.BatchNorm2d(6)

        self.conv0b = nn.Conv2d(
            6,
            6,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm0b = nn.BatchNorm2d(6)

        self.conv0c = nn.Conv2d(
            6,
            3,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm0c = nn.BatchNorm2d(3)

        self.fc01 = nn.Linear(150, 250 + 50, bias=False)
        self.batchnorm01 = nn.BatchNorm1d(3)
        self.resizenn1 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )

        self.conv1 = nn.Conv2d(
            6,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv1b = nn.Conv2d(
            16,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm1b = nn.BatchNorm2d(16)
        self.resizenn2 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv2b = nn.Conv2d(
            32,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm2b = nn.BatchNorm2d(32)

        self.resizenn3 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv3b = nn.Conv2d(
            64,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm3b = nn.BatchNorm2d(64)

        self.resizenn4 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )

        self.conv4 = nn.Conv2d(
            64,
            128,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv4b = nn.Conv2d(
            128,
            128,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm4b = nn.BatchNorm2d(128)

        self.resizenn5 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )

        self.conv5 = nn.Conv2d(
            128,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm5 = nn.BatchNorm2d(64)

        self.conv5b = nn.Conv2d(
            64,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm5b = nn.BatchNorm2d(64)

        self.conv5c = nn.Conv2d(
            64,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm5c = nn.BatchNorm2d(32)

        self.conv5d = nn.Conv2d(
            32,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm5d = nn.BatchNorm2d(32)

        self.conv5e = nn.Conv2d(
            32,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm5e = nn.BatchNorm2d(16)

        self.conv5f = nn.Conv2d(
            16,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.batchnorm5f = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(
            16,
            1,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )

        self.tanh4 = nn.Tanh()

        # PGA prediction
        self.fc_lcn = FCNC(n_vs=150, hidden_1=256, hidden_2=512)

    def forward(
        self,
        x: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------

        :param x             Noise vector.
        :param v1, v2, v3    Condition labels. The number of condition labels are fixed.
        """

        x = self.fc00(x)
        x = self.batchnorm00(x)

        # Change data dimension
        x = torch.unsqueeze(x, 3)

        # ----------- Condition labels  ------------
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)
        v3 = self.embed3(v3)

        # Reshape to concatenate with noise vector
        v1 = v1.view(-1, 1, 150, 1)
        v2 = v2.view(-1, 1, 150, 1)
        v3 = v3.view(-1, 1, 150, 1)

        x = torch.cat(
            [
                x,
                v1,
                v2,
                v3,
            ],
            dim=1,
        )

        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = F.relu(x)
        x = self.conv0b(x)
        x = self.batchnorm0b(x)
        x = F.relu(x)
        x = self.conv0c(x)
        x = self.batchnorm0c(x)
        x = F.relu(x)

        # Flatten the feature map
        x = torch.squeeze(x, 3)

        x = self.fc01(x)
        x = self.batchnorm01(x)
        x = F.relu(x)

        # Split data
        # For the PGA prediction
        xcn = x[:, :, 250:]
        xcn = xcn.reshape(-1, 3 * 50)

        # For the ground-motion generation
        x = x[:, :, :250]
        x = x.reshape(-1, 6, 125, 1)

        # Up-sampling by nearest_neighbor
        x = self.resizenn1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = self.batchnorm1b(x)
        x = F.relu(x)

        x = self.resizenn2(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = self.batchnorm2b(x)
        x = F.relu(x)

        x = self.resizenn3(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = self.batchnorm3b(x)
        x = F.relu(x)

        x = self.resizenn4(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.conv4b(x)
        x = self.batchnorm4b(x)
        x = F.relu(x)

        x = self.resizenn5(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.conv5b(x)
        x = self.batchnorm5b(x)
        x = F.relu(x)
        x = self.conv5c(x)
        x = self.batchnorm5c(x)
        x = F.relu(x)
        x = self.conv5d(x)
        x = self.batchnorm5d(x)
        x = F.relu(x)
        x = self.conv5e(x)
        x = self.batchnorm5e(x)
        x = F.relu(x)
        x = self.conv5f(x)
        x = self.batchnorm5f(x)
        x = F.relu(x)

        x = self.conv6(x)
        x_out = self.tanh4(x)

        xcn_out = self.fc_lcn(xcn)

        return (x_out, xcn_out)


if __name__ == "__main__":
    print("Initialize")
