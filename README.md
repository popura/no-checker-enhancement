# Checkerboard-Artifact-Free Image-Enhancement
This is an implementation for *Checkerboard-Artifact-Free Image-Enhancement Network Considering Local and Global Features*.
Note that this implementation is different from original codes
for the paper.

When you use this implementation for your research work,
please cite the paper.

The following is the bibtex entry.
```
@inproceedings{kinoshita2020checkerboard,
author = {Kinoshita, Yuma and Kiya, Hitoshi},
booktitle={Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
title={Checkerboard-Artifact-Free Image-Enhancement Network Considering Local and Global Features}, 
year={2020},
month = {Dec.},
pages={1139-1144},
url = {https://ieeexplore.ieee.org/document/9306349},
}
```

# Requirements
- Python 3.9 or later
- Poetry
- CUDA 10.2
- Pytorch 1.9 or later
- deepy 0.5.0 or later (in the `external` directory)
  Repo: https://github.com/popura/deepy-pytorch

For other requirements, see pyproject.toml

# Getting started
1. Clone this repository
    ```
    git clone https://github.com/popura/no-checker-enhancement.git
    cd no-checker-enhancement
    ```
1. Install requirements.

    By using poetry as a package manager, it is simply done by
    ```
    poetry install
    ```
1. Prepare a directory for storing a dataset (e.g., `./data/`)
1. Train a network.
   All outputs including trained models will be written in the `history` directory.
    ```
    poetry run python ./src/train.py
    ```
   For the first time, you can download Cai's dataset by running the following command
    ```
    poetry run python ./src/train.py dataset.download=True
    ```
1. Test the trained network.
   All outputs will be written in the `result` directory.
    ```
    poetry run python ./src/test.py
    ```

# Changing hyperparameters
For managing hyperparameters, we use the *hydra-core* package.
Default hyperparameters are defined in yaml files in the `./conf` directory.
You can overwrite the parameters when training a network.
For example, you can change the network architecture and the number of epochs by
```
poetry run python ./src/train.py model=unet epoch=200
```

