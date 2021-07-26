import time
import argparse
from pathlib import Path
import functools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F
from torchsummary import summary

import torchdataset
from torchtrain.trainer import AverageMeter

import kornia.losses

from omegaconf import OmegaConf, DictConfig

import model as mymodel
import dataset as mydataset
import util as myutil
import functional as myF
import layer as mylayer
import transform as mytransforms


def predict(data_dir, result_dir, net, device):
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)
    dataset = torchvision.datasets.ImageFolder(str(data_dir))
    #transforms = torchdataset.transform.Compose([mytransforms.ResizeToMultiple(divisor=16, interpolation=Image.BICUBIC),
    #                                             torchvision.transforms.ToTensor()])
    transforms = torchvision.transforms.ToTensor()
    to_pil = torchvision.transforms.ToPILImage()
    to_tensor = torchvision.transforms.ToTensor()

    with torch.no_grad():
        for i in range(len(dataset)):
            q, *_ = dataset.samples[i]
            q = Path(q)
            
            print('{:04d}/{:04d}: File Name {}'.format(i, len(dataset), q.name))

            sample, *_ = dataset[i]
            target = sample
            sample = transforms(sample).unsqueeze(0).to(device)
            predict = net(sample).to('cpu').clone().detach().squeeze(0)
            predict = to_pil(torch.clamp(predict, 0, 1))
            height = target.height
            width = target.width
            sample = to_pil(sample.to('cpu').clone().detach().squeeze(0))
            #sample = F.resize(sample, (height, width), Image.BICUBIC)
            #predict = F.resize(predict, (height, width), Image.BICUBIC)

            predict.save(str(p / ('{:04d}'.format(i) + '_' + q.stem + '_prediction.tiff')), compression=None)


def main(cfg: DictConfig, train_id: str, input_dir: str, output_dir: str) -> None:
    p = Path.cwd()
    myutil.print_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.param.activation == 'relu':
        activation = nn.ReLU
    elif cfg.model.param.activation == 'sin':
        activation = myF.Sin
    else:
        raise NotImplementedError

    cfg_down_layer = cfg.model.param.down_layer
    cfg_up_layer = cfg.model.param.up_layer
    if cfg_down_layer.name == "conv2d":
        down_conv_layer = nn.Conv2d
    elif cfg_down_layer.name == "down_sampling2d":
        down_conv_layer = functools.partial(mylayer.DownSampling2d, **cfg_down_layer.param)
    elif cfg_down_layer.name == "kernel_conv2d":
        down_conv_layer = functools.partial(mylayer.KernelConv2d, order=cfg_down_layer.param.order)
    if cfg.model.param.up_layer.name == "conv_transpose2d":
        up_conv_layer = nn.ConvTranspose2d
    elif cfg.model.param.up_layer.name == "up_sampling2d":
        up_conv_layer = functools.partial(mylayer.UpSampling2d, **cfg_up_layer.param)

    if cfg.model.name == 'unet':
        net = mymodel.UNet(n_channels=3, n_classes=3,
                           up_conv_layer=up_conv_layer,
                           down_conv_layer=down_conv_layer,
                           activation=activation)
    elif cfg.model.name == 'itm-net':
        net = mymodel.ITMNet(n_channels=3, n_classes=3,
                             up_conv_layer=up_conv_layer,
                             down_conv_layer=down_conv_layer,
                             activation=activation)
    elif cfg.model.name == 'seunet':
        net = mymodel.SEUNet(n_channels=3, n_classes=3,
                             up_conv_layer=up_conv_layer,
                             down_conv_layer=down_conv_layer,
                             activation=activation)
    elif cfg.model.name == 'se-itm-net':
        net = mymodel.SEITMNet(n_channels=3, n_classes=3,
                               up_conv_layer=up_conv_layer,
                               down_conv_layer=down_conv_layer,
                               activation=activation)
    else:
        raise NotImplementedError

    net.load_state_dict(torch.load(str(p / 'history' / train_id / 'trained_model.pth'),
                                   map_location=device))
    net.eval()
    net.to(device)
    summary(net, (3, 256, 256))

    data_dir = p / input_dir
    result_dir = p / output_dir / train_id
    predict(str(data_dir), str(result_dir), net, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default='./history',
                        help='Directory path for searching trained models')
    parser.add_argument('--input', type=str,
                        default='./predict/input',
                        help='Directory path for input images')
    parser.add_argument('--output', type=str,
                        default='./predict/output',
                        help='Directory path for output images')
    args = parser.parse_args()
    p = Path.cwd() / args.history_dir
    for q in p.glob('**/config.yaml'):
        cfg = OmegaConf.load(str(q))
        cfg.loader.num_workers = 0
        train_id = q.parent.name
        main(cfg, train_id, args.input, args.output)
