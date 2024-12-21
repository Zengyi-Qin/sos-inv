import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from dataset import FocusedWaveDataset, PlaneWaveDataset, ScanLineDataset
from model import SosNet
import argparse
import logging

def print_loss(loss_history, epoch, lr, logger):
    msg = [
        f'epoch {epoch}',
        f'lr {lr:.5f}',
    ]
    for k, v in loss_history.items():
        m = np.mean(v)
        msg.append(f'{k} {m:.3f}')
    msg = ', '.join(msg)
    print(msg)
    logger.info(msg)


def visualize(rf, sos, sos_pred, vis_dir, train_val, epoch):
    _, _, h, w = sos.shape
    rf = torch.nn.functional.interpolate(torch.abs(rf), (h, w), mode='bilinear')
    rf, _ = torch.max(rf, dim=1)
    rf = np.log(rf.detach().cpu().numpy() + 1e-6)

    sos = sos[:, 0].detach().cpu().numpy()
    sos_pred = sos_pred[:, 0].detach().cpu().numpy()

    def normalize(x):
        y = (x - x.min()) / (1e-9 + x.max() - x.min())
        y = (y * 255).astype(np.uint8)
        return y

    for i in range(len(sos)):
        rf_img = normalize(rf[i])
        sos_img = normalize(sos[i])
        sos_pred_img = normalize(sos_pred[i])

        black = np.zeros((8, rf_img.shape[1]), dtype=np.uint8)
        img = np.concatenate([rf_img, black, sos_img, black, sos_pred_img], axis=0)
        save_name = os.path.join(vis_dir, f'{train_val}_epoch_{epoch}_{i}.png')
        cv2.imwrite(save_name, img)
    

def train(model, dataloader, optimizer, device, epoch, vis_dir, logger):
    model.train()
    loss_history = {
        'loss': [],
        'mse': [],
        'mae': []
    }
    print(f'epoch {epoch + 1} starts')
    logger.info(f'epoch {epoch + 1} starts')
    for i, (rf, sos) in enumerate(dataloader):
        rf = rf.to(device)
        sos = sos.to(device)

        sos_pred = model(rf)
        loss = torch.mean((sos - sos_pred) ** 2) * 10

        mse = torch.mean((sos - sos_pred) ** 2) ** 0.5
        mae = torch.mean(torch.abs(sos - sos_pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history['loss'].append(loss.detach().cpu().numpy())
        loss_history['mse'].append(mse.detach().cpu().numpy())
        loss_history['mae'].append(mae.detach().cpu().numpy())

        if np.mod(i, 50) == 0:
            print_loss(loss_history, epoch+1, optimizer.param_groups[0]['lr'], logger)

    print_loss(loss_history, epoch+1, optimizer.param_groups[0]['lr'], logger)
    visualize(rf, sos, sos_pred, vis_dir, 'train', epoch)


def val(model, dataloader, device, epoch, vis_dir, logger):
    model.eval()
    loss_history = {
        'loss': [],
        'mse': [],
        'mae': []
    }
    print(f'epoch {epoch + 1} starts eval')
    logger.info(f'epoch {epoch + 1} starts eval')
    for i, (rf, sos) in enumerate(dataloader):
        rf = rf.to(device)
        sos = sos.to(device)

        sos_pred = model(rf)
        loss = torch.mean((sos - sos_pred) ** 2) * 10

        mse = torch.mean((sos - sos_pred) ** 2) ** 0.5
        mae = torch.mean(torch.abs(sos - sos_pred))

        loss_history['loss'].append(loss.detach().cpu().numpy())
        loss_history['mse'].append(mse.detach().cpu().numpy())
        loss_history['mae'].append(mae.detach().cpu().numpy())

    print_loss(loss_history, epoch+1, 0, logger)
    visualize(rf, sos, sos_pred, vis_dir, 'val', epoch)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SosNet(wide=True)
    model.to(device)

    model = torch.nn.DataParallel(
        model, device_ids=[i for i in range(torch.cuda.device_count())])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)],
        gamma=0.3,
    )

    save_dir = args.ckp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vis_dir = args.vis
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    if args.input_type == 'focused_wave':
        train_dataset = FocusedWaveDataset(
            os.path.join(args.data, 'train'), args.num_transmits)
        val_dataset = FocusedWaveDataset(
            os.path.join(args.data, 'val'), args.num_transmits)
    elif args.input_type == 'plane_wave':
        train_dataset = PlaneWaveDataset(
            os.path.join(args.data, 'train'), args.num_transmits)
        val_dataset = PlaneWaveDataset(
            os.path.join(args.data, 'val'), args.num_transmits)
    elif args.input_type == 'scan_line':
        train_dataset = ScanLineDataset(
            os.path.join(args.data, 'train'))
        val_dataset = ScanLineDataset(
            os.path.join(args.data, 'val'))
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers,
        persistent_workers=True
    )

    log_path = os.path.join(args.ckp, 'logs.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logger = logging.getLogger()

    for i in range(args.epochs):
        train(model, train_loader, optimizer, device, i, vis_dir, logger)
        val(model, val_loader, device, i, vis_dir, logger)
        scheduler.step()
        if np.mod(i + 1, 20) == 0 or i + 1 == args.epochs:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir, 
                    "{}_{}_trn_{}_epoch.pth".format(args.input_type, args.num_transmits, str(i+1).zfill(3))),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_type", type=str, default='focused_wave', choices=['focused_wave', 'plane_wave', 'scan_line']
    )
    parser.add_argument(
        "--num_transmits", type=int, default=3, choices=[1, 3, 5, 7, 9, 11]
    )
    parser.add_argument(
        "--batchsize", type=int, default=8
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="number of workers in dataloader"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="number of epochs in training"
    )
    parser.add_argument("--data", default="./data", help="dataset root")
    parser.add_argument(
        "--ckp", default="./outputs/checkpoint", help="path to save checkpoints"
    )
    parser.add_argument(
        "--vis", default="./outputs/vis", help="path to save visualizations"
    )
    args = parser.parse_args()
    main(args)
