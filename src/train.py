from pathlib import Path
from typing import Optional

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary

from omegaconf import OmegaConf, DictConfig
import hydra

import deepy
from deepy.data.vision import CaiMEImageDataset
from deepy.train.trainer import RegressorTrainer
from deepy.train.extension import (
    IntervalTrigger,
    MinValueTrigger,
    ModelSaver,
    HistorySaver
)

import util as myutil
import train_id as tid



def save_model(model: nn.Module, path: str) -> None:
    """Save a DNN model (torch.nn.Module).
    Args:
        model: torch.nn.Module object
        path: Directory path where model will be saved
    Returns:
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), path)


def get_transform(cfg: DictConfig):
    pre_transforms = None

    dict_cfg_crop = cfg.dataset.transforms.paired_random_resized_crop.param
    if dict_cfg_crop.interpolation == 'Image.BICUBIC':
        interpolation = Image.BICUBIC

    transforms = deepy.data.transform.PairedCompose([
        deepy.data.transform.ToPairedTransform(
            torchvision.transforms.ToTensor()),
        deepy.data.vision.transform.PairedRandomHorizontalFlip(
            **cfg.dataset.transforms.paired_random_horizontal_flip.param),
        deepy.data.vision.transform.PairedRandomResizedCrop(
            size=tuple(dict_cfg_crop.size), scale=tuple(dict_cfg_crop.scale),
            ratio=tuple(dict_cfg_crop.ratio), interpolation=interpolation)
    ])

    return pre_transforms, transforms


def get_data_loaders(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    p = Path(cwd) / cfg.dataset.path

    _, transforms = get_transform(cfg)

    trainset = CaiMEImageDataset(
        root=str(p),
        train=True,
        transforms=transforms,
        pre_load=cfg.dataset.pre_load,
        download=cfg.dataset.download
    )
    
    valset = CaiMEImageDataset(
        root=str(p),
        train=False,
        transforms=deepy.data.transform.PairedCompose([
            deepy.data.transform.ToPairedTransform(
                torchvision.transforms.Resize(1080)),
            deepy.data.transform.ToPairedTransform(
                deepy.data.vision.transform.ResizeToMultiple(2**cfg.model.param.depth)),
            deepy.data.transform.ToPairedTransform(
                torchvision.transforms.ToTensor())
        ]),
        pre_load=cfg.dataset.pre_load,
        download=cfg.dataset.download
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers)

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.loader.num_workers)
    
    return trainloader, valloader


def get_optimizer(params, cfg: DictConfig):
    if cfg.optimizer.name == "sgd":
        return optim.SGD(params, **cfg.optimizer.params)
    elif cfg.optimizer.name == "adam":
        return optim.Adam(params, **cfg.optimizer.params)
    else:
        raise NotImplementedError


def get_scheduler(optimizer, cfg: DictConfig):
    if cfg.lr_scheduler.name == "multi_step":
        return optim.lr_scheduler.MultiStepLR(optimizer, **cfg.lr_scheduler.params)
    elif cfg.lr_scheduler.name is None:
        return None
    else:
        raise NotImplementedError


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    cwd = Path(hydra.utils.get_original_cwd())

    myutil.print_config(cfg)

    # Setting history directory
    # All outputs will be written into (p / "history" / train_id).
    train_id = tid.generate_train_id(cfg)
    history_dir = cwd / "history" / train_id
    if not history_dir.exists():
        history_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = history_dir / "config.yaml"
    if cfg_path.exists():
        existing_cfg = OmegaConf.load(str(history_dir / "config.yaml"))
        if not myutil.is_same_config(cfg, existing_cfg):
            raise ValueError("Train ID {} already exists, but config is different".format(train_id))

    # Saving cfg
    OmegaConf.save(cfg, str(history_dir / "config.yaml"))

    # Setting seed 
    if cfg.seed is not None:
        myutil.set_random_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    trainloaders, valloaders = get_data_loaders(cfg)
    net = myutil.get_model(cfg)
    if device.type == "cuda":
        net = torch.nn.DataParallel(net)
    net = net.to(device)
    summary(net, input_size=(1, 3, 256, 256))

    criterion = nn.L1Loss()
    optimizer = get_optimizer(net.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)
    extensions = [ModelSaver(directory=history_dir,
                             name=lambda x: cfg.model.name+"_best.pth",
                             trigger=MinValueTrigger(mode="validation", key="loss")),
                  HistorySaver(directory=history_dir,
                               name=lambda x: cfg.model.name+"_history.pth",
                               trigger=IntervalTrigger(period=1))]

    trainer = RegressorTrainer(net, optimizer, criterion, trainloaders,
                               scheduler=scheduler, extensions=extensions,
                               init_epoch=0,
                               device=device)
    trainer.train(cfg.epoch, valloaders)

    save_model(net, str(history_dir / "{}.pth".format(cfg.model.name)))


if __name__ == "__main__":
    main()
