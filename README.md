# PSHA Using a Ground-Motion Generative Model (GMGM)

This repository contains the code and hyperparameters for the paper:

**Waveform-Based Probabilistic Seismic Hazard Analysis Using Ground-Motion Generative Models**\
Yuma Matsumoto, Taro Yaoyama, Sangwon Lee, Asako Iwaki, and Tatsuya Itoi\
Bulletin of the Seismological Society of America (2026)\
https://doi.org/10.1785/0120250269

Please cite the paper if you use this repository as part of a published research project.

## Operating Environment

The code in this repository has been tested and is known to run in the following environment:

- Ubuntu 24.04.3 LTS
- Python 3.12.3

The required Python dependencies are listed in `./pyproject.toml`.
You can reproduce the virtual environment with `uv` as follows.

### Reproducing the Python environment with `uv`

> Prerequisite: `uv` must be installed on your system if you choose to create the virtual environment using `uv`.
> See to [the official guide](https://docs.astral.sh/uv/getting-started/installation/).

Clone this repository to any directory:

```bash
git clone https://github.com/Mat-main-00/psha-gmgm.git
cd psha-gmgm
```

Create a Python 3.12 virtual environment:

```bash
uv venv --python 3.12
```

Install the dependencies:

```bash
uv sync
```

This will create a virtual environment at `psha-gmgm/.venv/` with all required libraries installed.
Run Python files inside the environment as follows:

```bash
uv run <file_to_execute>.py
```

## Usage

The structure of the code included in this repository is summarized below:

```bash
.
├── data
├── out_data
│   ├── csgmgm
│   ├── cwgmgm
│   └── sgmgm
├── pkgs
│   └── src
│       └── pkgs
│           ├── models
│           │   ├── csgmgm    <- DNN for CS-GMGM
│           │   ├── cwgmgm    <- DNN for CW-GMGM
│           │   └── sgmgm     <- DNN for S-GMGM
│           ├── optimal       <- Determination of the optimal number of epochs
│           └── util          <- Utility functions required for running PSHA
├── psha_sampling             <- PSHA execution using GMGM
└── training                  <- Training of the GMGM
```

## Dataset Preparation

Most of the data used in this study cannot be redistributed. Accordingly, the files under `data/` in this repository are dummies.
To train the GMGM, you must prepare the dataset in advance.

- Replace the contents of each `dataset_*.csv` with your own dataset.
- Header fields of `dataset_*.csv`:
    - `file_name`: Path to the `.npy` file containing amplitude-normalized ground-motion data.
    - `log10_pga`: Common logarithm of the Peak Ground Acceleration (PGA) of the corresponding ground‑motion data.
    - `mw`: Moment magnitude, $$M_W$$
    - `log10_v30`: Common logarithm of $$V\_{\\mathrm{S}30}$$ (original unit: m/s)
    - `log_fault_dist`: Natural logarithm of rupture distance, $$R\_{\\mathrm{RUP}}$$ (original unit: km)
- Structure of `example_*.npy` files:
    - Each file contains a one-dimensional array of ground-motion time-history data.
    - The array length equals the number of time steps in the record.

## License

This code is licensed under the MIT License.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
