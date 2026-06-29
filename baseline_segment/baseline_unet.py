#!/usr/bin/env python3
"""Direct PZ/CG 3D UNet baseline using the segment dataloader.

Label convention: background=0, PZ=1, CG=2, whole prostate=(label > 0).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
SEGMENT_DIR = ROOT_DIR / 'segment'
if str(SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_DIR))

from dataloader3d import dataset_loaders


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_channels=24):
        super().__init__()
        c = int(base_channels)
        self.enc1 = DoubleConv3D(in_channels, c)
        self.enc2 = DoubleConv3D(c, c * 2)
        self.enc3 = DoubleConv3D(c * 2, c * 4)
        self.enc4 = DoubleConv3D(c * 4, c * 8)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(c * 2, c)
        self.out = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4)
        d3 = self._match_and_concat(d3, e3)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self._match_and_concat(d2, e2)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self._match_and_concat(d1, e1)
        d1 = self.dec1(d1)
        return self.out(d1)

    @staticmethod
    def _match_and_concat(x, skip):
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)


def setup_device(gpu_id):
    return 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() and int(gpu_id) >= 0 else 'cpu'


def get_dataloader(data_root, path, phase, batch_size, crop_size=None, spacing=None, shuffle=False):
    dataset = dataset_loaders(
        os.path.join(data_root, path),
        phase,
        batch_size=batch_size,
        np_var='vol',
        add_batch_axis=True,
        pad_shape=None,
        resize_factor=spacing,
        crop_size=crop_size,
        istest=(phase != 'train'),
        transform=None,
        ifbin=False,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (probs * one_hot).sum(dim=dims)
    den = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice[1:].mean()


def segmentation_loss(logits, target, ce_weight=1.0, dice_weight=1.0):
    ce = F.cross_entropy(logits, target.long())
    dl = dice_loss(logits, target)
    return ce_weight * ce + dice_weight * dl, ce.detach(), dl.detach()


def dice_binary(pred, target, eps=1e-6):
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    return float((2.0 * np.logical_and(pred, target).sum() + eps) / (pred.sum() + target.sum() + eps))


def hd95_binary(pred, target, spacing=(1, 1, 1), empty_value=100.0):
    pred_points = np.argwhere(np.asarray(pred) > 0)
    target_points = np.argwhere(np.asarray(target) > 0)
    if pred_points.size == 0 or target_points.size == 0:
        return float(empty_value)
    spacing = np.asarray(spacing, dtype=np.float32)
    pred_points = pred_points * spacing
    target_points = target_points * spacing
    tree_pred = KDTree(pred_points)
    tree_target = KDTree(target_points)
    distances = np.concatenate([
        tree_target.query(pred_points)[0],
        tree_pred.query(target_points)[0],
    ])
    return float(np.percentile(distances, 95))


def metrics_for_case(pred, label, spacing):
    return {
        'PZ Dice': dice_binary(pred == 1, label == 1),
        'CG Dice': dice_binary(pred == 2, label == 2),
        'Prostate Dice': dice_binary(pred > 0, label > 0),
        'PZ Dist': hd95_binary(pred == 1, label == 1, spacing=spacing),
        'CG Dist': hd95_binary(pred == 2, label == 2, spacing=spacing),
        'Prostate Dist': hd95_binary(pred > 0, label > 0, spacing=spacing),
    }


def save_visual(image, pred, label, path):
    mask = (pred > 0) | (label > 0)
    z = int(np.median(np.where(mask)[-1])) if mask.sum() > 0 else image.shape[-1] // 2
    image_slice = image[..., z]
    image_slice = image_slice - image_slice.min()
    if image_slice.max() > 0:
        image_slice = image_slice / image_slice.max()
    concat = np.concatenate([
        (255 * np.rot90(image_slice, k=3)).astype(np.uint8),
        (127 * np.rot90(pred[..., z].astype(np.uint8), k=3)).astype(np.uint8),
        (127 * np.rot90(label[..., z].astype(np.uint8), k=3)).astype(np.uint8),
    ], axis=1)
    cv2.imwrite(str(path), concat)


def affine_from_batch(data):
    affine = data.get('affine')
    if torch.is_tensor(affine):
        return affine[0].cpu().numpy()
    return np.asarray(affine[0])


def train(args):
    device = setup_device(args.gpus)
    save_dir = Path(args.save_dir) / 'unet3d_{}'.format(args.path)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / 'config.json').open('w') as f:
        json.dump(vars(args), f, indent=2)

    train_loader = get_dataloader(args.data_root, args.path, 'train', args.batch_size, args.crop_size, args.spacing, shuffle=True)
    val_loader = get_dataloader(args.data_root, args.path, 'val', 1, args.crop_size, args.spacing, shuffle=False)
    model = UNet3D(args.inch, args.outch, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for i, data in enumerate(train_loader):
            inputs = data['img'].to(device).float()
            labels = data['seg'].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss, ce, dl = segmentation_loss(logits, labels, args.ce_loss_weight, args.dice_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if i % args.print_freq == 0:
                print('epoch {}, batch {}, loss={:.4f}, ce={:.4f}, dice_loss={:.4f}'.format(epoch, i, float(loss), float(ce), float(dl)))

        val_metrics = evaluate(args, model, val_loader, device, save_dir=None)
        mean_loss = float(np.mean(losses)) if losses else float('inf')
        score = val_metrics['Average']['Mean Foreground Dice']
        print('epoch {}, train_loss={:.4f}, val_pz={:.4f}, val_cg={:.4f}, val_prostate={:.4f}'.format(
            epoch, mean_loss, val_metrics['Average']['PZ Dice'], val_metrics['Average']['CG Dice'], val_metrics['Average']['Prostate Dice']
        ))
        state = {'model': model.state_dict(), 'epoch': epoch, 'metrics': val_metrics, 'args': vars(args)}
        torch.save(state, save_dir / 'latest.pth')
        if score > best_score:
            best_score = score
            torch.save(state, save_dir / 'best.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(state, save_dir / 'epoch_{}.pth'.format(epoch))


def evaluate(args, model, loader, device, save_dir=None):
    model.eval()
    rows = []
    metrics = {}
    visual_dir = None
    if save_dir is not None:
        visual_dir = Path(save_dir) / '{}_visual'.format(args.path)
        visual_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data['img'].to(device).float()
            labels_t = data['seg'].to(device).long()
            logits = model(inputs)
            pred_t = torch.argmax(logits, dim=1)
            pred = pred_t[0].cpu().numpy().astype(np.uint8)
            label = labels_t[0].cpu().numpy().astype(np.uint8)
            key = data['key'][0] if isinstance(data['key'], (list, tuple)) else str(data['key'])
            case_metrics = metrics_for_case(pred, label, args.spacing)
            metrics[key] = case_metrics
            rows.append([
                case_metrics['PZ Dice'], case_metrics['CG Dice'], case_metrics['Prostate Dice'],
                case_metrics['PZ Dist'], case_metrics['CG Dist'], case_metrics['Prostate Dist'],
            ])
            print('{}: {}, PZ Dice={:.4f}, CG Dice={:.4f}, prostate Dice={:.4f}, PZ Dist={:.4f}, CG Dist={:.4f}, prostate Dist={:.4f}'.format(
                i, key, case_metrics['PZ Dice'], case_metrics['CG Dice'], case_metrics['Prostate Dice'],
                case_metrics['PZ Dist'], case_metrics['CG Dist'], case_metrics['Prostate Dist']
            ))
            if visual_dir is not None:
                affine = affine_from_batch(data)
                nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), visual_dir / '{}_pred.nii.gz'.format(key))
                image = data['img'][0, 0].cpu().numpy().astype(np.float32)
                save_visual(image, pred, label, visual_dir / '{}.png'.format(key))

    rows = np.asarray(rows, dtype=np.float32)
    mean = rows.mean(axis=0) if len(rows) else np.zeros(6, dtype=np.float32)
    std = rows.std(axis=0) if len(rows) else np.zeros(6, dtype=np.float32)
    metrics['Average'] = {
        'PZ Dice': float(mean[0]),
        'CG Dice': float(mean[1]),
        'Prostate Dice': float(mean[2]),
        'PZ Dist': float(mean[3]),
        'CG Dist': float(mean[4]),
        'Prostate Dist': float(mean[5]),
        'Mean Foreground Dice': float((mean[0] + mean[1]) / 2.0),
        'Std': {
            'PZ Dice': float(std[0]), 'CG Dice': float(std[1]), 'Prostate Dice': float(std[2]),
            'PZ Dist': float(std[3]), 'CG Dist': float(std[4]), 'Prostate Dist': float(std[5]),
        },
    }
    return metrics


def test(args):
    device = setup_device(args.gpus)
    save_dir = Path(args.save_dir) / args.load_from_dir
    checkpoint = Path(args.checkpoint) if args.checkpoint else save_dir / 'best.pth'
    model = UNet3D(args.inch, args.outch, args.base_channels).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
    loader = get_dataloader(args.data_root, args.path, 'test', 1, args.crop_size, args.spacing, shuffle=False)
    metrics = evaluate(args, model, loader, device, save_dir=save_dir if args.save_output else None)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / '{}_baseline_unet_metrics.json'.format(args.path)).open('w') as f:
        json.dump(metrics, f, indent=2)
    avg = metrics['Average']
    print('Baseline UNet: PZ Dice={:.4f}, CG Dice={:.4f}, prostate Dice={:.4f}, PZ Dist={:.4f}, CG Dist={:.4f}'.format(
        avg['PZ Dice'], avg['CG Dice'], avg['Prostate Dice'], avg['PZ Dist'], avg['CG Dist']
    ))


def parse_args():
    parser = argparse.ArgumentParser(description='Direct 3D UNet baseline for PZ/CG segmentation')
    parser.add_argument('--data_root', type=str, default='../../Datasets/ProstateDatasets/data_split_files')
    parser.add_argument('--path', type=str, default='4-prostate158')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--spacing', type=float, nargs='+', default=[0.5, 0.5, 1.0])
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--inch', type=int, default=1)
    parser.add_argument('--outch', type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ce_loss_weight', type=float, default=1.0)
    parser.add_argument('--dice_loss_weight', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--load_from_dir', type=str, default='unet3d_4-prostate158')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--gpus', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'train':
        train(args)
    else:
        test(args)
