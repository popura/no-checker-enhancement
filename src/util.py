import functools
from pathlib import Path

import numpy as np
from PIL import Image

from omegaconf import DictConfig

import torch
import torch.nn as nn
import torchvision


import dataset as mydataset
import model as mymodel
import transform as mytransforms


def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(cfg.pretty())
    print('--------------------')


def is_same_config(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    """Compare cfg1 with cfg2.

    Args:
        cfg1: Config
        cfg2: Config

    Returns:
        True if cfg1 == cfg2 else False

    """
    return cfg1 == cfg2


def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(cfg.pretty())
    print('--------------------')


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(cfg: DictConfig) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_down_layer = cfg.model.param.down_layer
    if cfg_down_layer.name == "conv2d":
        down_conv_layer = nn.Conv2d
    elif cfg_down_layer.name == "down_sampling2d":
        down_conv_layer = functools.partial(mylayer.DownSampling2d, **cfg_down_layer.param)
    elif cfg_down_layer.name == "kernel_conv2d":
        down_conv_layer = functools.partial(mylayer.KernelConv2d, order=cfg_down_layer.param.order)


    cfg_up_layer = cfg.model.param.up_layer
    if cfg.model.param.up_layer.name == "conv_transpose2d":
        up_conv_layer = nn.ConvTranspose2d
    elif cfg.model.param.up_layer.name == "up_sampling2d":
        up_conv_layer = functools.partial(mylayer.UpSampling2d, **cfg_up_layer.param)


    if cfg.model.name == "unet":
        net = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            up_conv=up_conv_layer,
            down_conv=down_conv_layer,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
            final_activation=nn.ReLU)
    elif cfg.model.name == "itmnet":
        net = ITMNet(
            in_channels=3,
            out_channels=3,
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            up_conv=up_conv_layer,
            down_conv=down_conv_layer,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
            final_activation=nn.ReLU)
    elif cfg.model.name == "seunet":
        net = SEUNet(
            in_channels=3,
            out_channels=3,
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            up_conv=up_conv_layer,
            down_conv=down_conv_layer,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
            final_activation=nn.ReLU)
    elif cfg.model.name == "se-itm-net":
        net = SEITMNet(
            in_channels=3,
            out_channels=3,
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            up_conv=up_conv_layer,
            down_conv=down_conv_layer,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
            final_activation=nn.ReLU)
    else:
        raise NotImplementedError()

    return net
