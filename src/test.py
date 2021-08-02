import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as F
from torchinfo import summary

import kornia.losses

from omegaconf import OmegaConf, DictConfig

import deepy
from deepy.data.vision import CaiMEImageDataset

import util as myutil



def get_data_loader(cfg: DictConfig):
    p = Path.cwd() / cfg.dataset.path

    testset = CaiMEImageDataset(
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

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers)
    
    return testloader


def predict(path, dataset, net, device):
    p = Path(path)
    (p / 'imgs').mkdir(parents=True, exist_ok=True)

    psnr = kornia.losses.PSNRLoss(max_val=1)
    ssim = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
    raw_df = pd.DataFrame(columns=['PSNR', 'SSIM'])
    to_pil = torchvision.transforms.ToPIL()

    with torch.no_grad():
        for i in range(len(dataset)):
            q, *_ = dataset.samples[i]
            q = Path(q)
            
            print('{:04d}/{:04d}: File Name {}'.format(i, len(dataset), q.name))

            sample, target = dataset[i]
            sample = sample.unsqueeze(0).to(device)
            predict = net(sample).to('cpu').clone().detach()
            sample = sample.to('cpu').clone().detach()

            psnr_loss = psnr(predict, target)
            ssim_loss = ssim(predict, target)
            raw_df = raw_df.append({'PSNR': psnr_loss.item(),
                                    'SSIM': ssim_loss.item()},
                                   ignore_index=True)
            
            to_pil(sample.squeeze(0)).save(str(p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_input.png')))
            to_pil(predict.squeeze(0)).save(str(p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_prediction.png')))
            to_pil(target.squeeze(0)).save(str(p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_target.png')))

    summary_df = pd.DataFrame([raw_df.mean(), raw_df.std()], index=['mean', 'std']) 
    raw_df.to_csv(str(p / 'raw_result.csv'))
    summary_df.to_csv(str(p / 'summary.csv'))
    print(summary_df)


def main(cfg: DictConfig, train_id: str) -> None:
    p = Path.cwd()
    myutil.print_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_data_loader(cfg)
    net = myutil.get_model(cfg)
    net.load_state_dict(torch.load(str(p / 'history' / train_id / f'{cfg.model.name}_best.pth'),
                                   map_location=device))
    net.eval()
    net.to(device)
    summary(net, (1, 3, 256, 256))

    result_dir = p / 'result' / train_id
    predict(str(result_dir), loader.dataset, net, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default='./history',
                        help='Directory path for searching trained models')
    args = parser.parse_args()
    p = Path.cwd() / args.history_dir
    for q in p.glob('**/config.yaml'):
        cfg = OmegaConf.load(str(q))
        cfg.loader.num_workers = 0
        train_id = q.parent.name
        main(cfg, train_id)
