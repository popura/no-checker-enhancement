# iTMNet
This is an implementation for *Checkerboard-Artifact-Free Image-Enhancement Network Considering Local and Global Features*.

When you use this implementation for your research work,
please cite the paper.

The following is the bibtex entry.
```
@inproceedings{kinoshita2020checkerboard,
author = {Kinoshita, Yuma and Kiya, Hitoshi},
booktitle={2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
title={Checkerboard-Artifact-Free Image-Enhancement Network Considering Local and Global Features}, 
year={2020},
month = {Dec.},
pages={1139-1144},
url = {https://ieeexplore.ieee.org/document/9306349},
}
```

# Requirements
- Python 3.9 or later
- Pytorch 1.8 or later
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

    If you use poetry as a package manager, it is done by
    ```
    poetry install
    ```
1. Prepare a directory for storing HDR images (e.g., `./data/HDRForCNN/`), where the directory should have `train`, `validation`, and `test` directories.
1. Put HDR images into the `train`, `validation`, and `test` directories.
1. Rewrite the path to the data directory in `./conf/dataset/mydataset.yaml`
1. Train iTM-Net.
   All outputs including trained models will be written in the `history` directory.
    ```
    poetry run python ./src/train.py
    ```
1. Test.
   All outputs will be written in the `result` directory.
    ```
    poetry run python ./src/test.py
    ```
