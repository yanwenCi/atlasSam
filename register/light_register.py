#!/usr/bin/env python3
"""Lightweight atlas-to-target registration training.

This script trains a small pairwise registration network that warps one atlas image/mask
onto target cases. Two transform heads are supported:

  affine: predicts a 3x4 affine matrix initialized at identity.
  ddf: predicts a sparse displacement grid, e.g. 10x10x10, upsampled to image size.

The loss is image similarity plus smoothness, with optional prostate/gland mask
supervision from labels such as prostate_zones.nii.gz.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = ('.nii.gz', '.nii', '.mgz')
DEFAULT_ATLAS_BANK_DATA_ROOT = (
    '/cluster/project7/longitude/Datasets/ProstateDatasets/3-picai-data/data-ROI-192-96'
)
LEGACY_ATLAS_BANK_ROOTS = (
    '/raid/candi/Wen/Dataset/ProstateDatasets/3-picai-data/data-ROI-192-96',
    '/raid/candi/Wen/ProstateSeg/Data/scripts/3-picai-data/data-ROI-192-96',
)


def resolve_path(value, base_dir=None, data_root=None):
    if value is None or value == '' or value == 'None':
        return None
    path = Path(value)
    if path.exists():
        return path
    if base_dir is not None and not path.is_absolute():
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate
    text = str(path)
    if data_root:
        for legacy_root in LEGACY_ATLAS_BANK_ROOTS:
            if text.startswith(legacy_root):
                candidate = Path(data_root) / Path(text).relative_to(legacy_root)
                if candidate.exists():
                    return candidate
        case_parts = [part for part in path.parts if part.startswith('P-')]
        if case_parts:
            candidate = Path(data_root) / case_parts[-1] / path.name
            if candidate.exists():
                return candidate
    return path


def split_file_from_args(args):
    candidates = [
        Path(args.data_root) / args.path / '{}.txt'.format(args.phase),
        Path(args.data_root) / args.path / 'path_list.txt',
        Path(args.data_root) / '{}.txt'.format(args.phase),
        Path(args.data_root) / 'path_list.txt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not find split file. Tried: {}'.format(', '.join(str(p) for p in candidates)))


def atlas_bank_data_root(args, bank_dir):
    metadata_path = Path(bank_dir) / 'bank_metadata.json'
    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
            data_root = metadata.get('data_root', '')
            if data_root and Path(data_root).exists():
                return data_root
        except Exception as exc:
            print('Warning: could not read atlas-bank metadata: {}'.format(exc))
    return args.atlas_bank_data_root or DEFAULT_ATLAS_BANK_DATA_ROOT


def select_atlas_from_bank(args):
    bank_dir = Path(args.atlas_bank_dir)
    reference_csv = bank_dir / 'reference_bank.csv'
    if not reference_csv.exists():
        raise FileNotFoundError('Atlas bank CSV not found: {}'.format(reference_csv))
    data_root = atlas_bank_data_root(args, bank_dir)
    with reference_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get('case_id', '')
            if args.atlas_case_id and case_id != args.atlas_case_id:
                continue
            t2_path = resolve_path(row.get('bank_t2_path') or row.get('t2_path'), bank_dir, data_root)
            mask_path = resolve_path(row.get('bank_zones_path') or row.get('zones_path') or row.get('prostate_zones_path'), bank_dir, data_root)
            if t2_path is not None and mask_path is not None and t2_path.exists() and mask_path.exists():
                print('Using atlas {} from {}'.format(case_id or t2_path.parent.name, reference_csv))
                return str(t2_path), str(mask_path)
    raise RuntimeError('No readable atlas found in {}'.format(reference_csv))


def load_nifti(path):
    nii = nib.load(str(path))
    return nii.get_fdata().astype(np.float32), nii.affine


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)
    mask = image > 0
    voxels = image[mask] if mask.any() else image.reshape(-1)
    mean = float(voxels.mean())
    std = float(voxels.std())
    if std < 1e-6:
        std = 1.0
    return (image - mean) / std


def resize_tensor(x, shape, mode):
    if tuple(x.shape[-3:]) == tuple(shape):
        return x
    kwargs = {} if mode == 'nearest' else {'align_corners': False}
    return F.interpolate(x, size=tuple(shape), mode=mode, **kwargs)


def make_base_grid(shape, device):
    d, h, w = shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, d, device=device),
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing='ij',
    )
    return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)


def warp_with_ddf(volume, ddf):
    # ddf is Bx3xDxHxW in normalized grid coordinates.
    grid = make_base_grid(volume.shape[-3:], volume.device)
    grid = grid + ddf.permute(0, 2, 3, 4, 1)
    return F.grid_sample(volume, grid, mode='bilinear', padding_mode='border', align_corners=True)


def warp_mask_with_ddf(mask, ddf):
    grid = make_base_grid(mask.shape[-3:], mask.device)
    grid = grid + ddf.permute(0, 2, 3, 4, 1)
    return F.grid_sample(mask, grid, mode='nearest', padding_mode='zeros', align_corners=True)


def ncc_loss(a, b, eps=1e-6):
    a = a - a.mean(dim=(2, 3, 4), keepdim=True)
    b = b - b.mean(dim=(2, 3, 4), keepdim=True)
    num = (a * b).mean(dim=(2, 3, 4))
    den = torch.sqrt((a.square().mean(dim=(2, 3, 4)) * b.square().mean(dim=(2, 3, 4))).clamp_min(eps))
    return 1.0 - (num / den).mean()


def dice_loss(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    dims = tuple(range(2, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    den = pred.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * inter + eps) / (den + eps)).mean()


def ddf_smoothness(ddf):
    dz = (ddf[:, :, 1:] - ddf[:, :, :-1]).square().mean()
    dy = (ddf[:, :, :, 1:] - ddf[:, :, :, :-1]).square().mean()
    dx = (ddf[:, :, :, :, 1:] - ddf[:, :, :, :, :-1]).square().mean()
    return (dx + dy + dz) / 3.0


class AtlasTargetDataset(Dataset):
    def __init__(self, split_file, atlas_t2, atlas_mask, data_root=None, image_size=None, mask_threshold=1.0, max_cases=None, skip_case_id=''):
        self.split_file = Path(split_file)
        rows = [line.strip().split() for line in self.split_file.read_text().splitlines() if line.strip()]
        self.data_root = data_root
        filtered = []
        for row in rows:
            t2_path, _ = self._target_paths(row, data_root=data_root)
            if skip_case_id and Path(t2_path).parent.name == skip_case_id:
                continue
            filtered.append(row)
        if max_cases:
            filtered = filtered[:max_cases]
        self.rows = filtered
        self.image_size = tuple(image_size) if image_size else None
        self.mask_threshold = float(mask_threshold)

        atlas_img, _ = load_nifti(atlas_t2)
        atlas_msk, _ = load_nifti(atlas_mask)
        self.atlas_img = normalize_image(atlas_img)
        self.atlas_mask = (atlas_msk > self.mask_threshold).astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def _target_paths(self, row, data_root=None):
        t2_path = resolve_path(row[0], data_root=data_root or self.data_root)
        if len(row) >= 4:
            mask_path = resolve_path(row[3], data_root=data_root or self.data_root)
        else:
            mask_path = Path(str(t2_path).replace('t2.nii', 'prostate_zones.nii'))
        return str(t2_path), str(mask_path)

    def __getitem__(self, idx):
        row = self.rows[idx]
        t2_path, mask_path = self._target_paths(row)
        fixed_img, affine = load_nifti(t2_path)
        fixed_mask, _ = load_nifti(mask_path)
        fixed_img = normalize_image(fixed_img)
        fixed_mask = (fixed_mask > self.mask_threshold).astype(np.float32)

        fixed = torch.from_numpy(fixed_img)[None, None]
        fixed_mask_t = torch.from_numpy(fixed_mask)[None, None]
        moving = torch.from_numpy(self.atlas_img)[None, None]
        moving_mask = torch.from_numpy(self.atlas_mask)[None, None]

        target_shape = self.image_size or tuple(fixed.shape[-3:])
        fixed = resize_tensor(fixed, target_shape, 'trilinear')[0]
        fixed_mask_t = resize_tensor(fixed_mask_t, target_shape, 'nearest')[0]
        moving = resize_tensor(moving, target_shape, 'trilinear')[0]
        moving_mask = resize_tensor(moving_mask, target_shape, 'nearest')[0]

        return {
            'fixed': fixed.float(),
            'fixed_mask': fixed_mask_t.float(),
            'moving': moving.float(),
            'moving_mask': moving_mask.float(),
            'case_id': Path(t2_path).parent.name,
            'affine': torch.from_numpy(affine.astype(np.float32)),
        }


class LightRegistrationNet(nn.Module):
    def __init__(self, transform='ddf', grid_size=(10, 10, 10), max_disp=0.20):
        super().__init__()
        self.transform = transform
        self.grid_size = tuple(grid_size)
        self.max_disp = float(max_disp)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, stride=2, padding=1), nn.InstanceNorm3d(8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.InstanceNorm3d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.InstanceNorm3d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        out_dim = 12 if transform == 'affine' else 3 * int(np.prod(self.grid_size))
        self.head = nn.Linear(32, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, moving, fixed):
        feat = self.encoder(torch.cat([moving, fixed], dim=1)).flatten(1)
        raw = self.head(feat)
        if self.transform == 'affine':
            delta = raw.view(-1, 3, 4)
            identity = torch.zeros_like(delta)
            identity[:, 0, 0] = 1.0
            identity[:, 1, 1] = 1.0
            identity[:, 2, 2] = 1.0
            theta = identity + 0.05 * delta
            grid = F.affine_grid(theta, fixed.shape, align_corners=True)
            warped = F.grid_sample(moving, grid, mode='bilinear', padding_mode='border', align_corners=True)
            return warped, theta, None
        sparse = torch.tanh(raw.view(-1, 3, *self.grid_size)) * self.max_disp
        ddf = F.interpolate(sparse, size=fixed.shape[-3:], mode='trilinear', align_corners=True)
        warped = warp_with_ddf(moving, ddf)
        return warped, None, ddf


def warp_moving_mask(model, moving_mask, fixed, theta, ddf):
    if model.transform == 'affine':
        grid = F.affine_grid(theta, fixed.shape, align_corners=True)
        return F.grid_sample(moving_mask, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    return warp_mask_with_ddf(moving_mask, ddf)


def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    image_size = args.image_size if args.image_size else None
    split_file = split_file_from_args(args)
    atlas_t2, atlas_mask = select_atlas_from_bank(args)
    atlas_case_id = Path(atlas_t2).parent.name
    dataset = AtlasTargetDataset(
        split_file,
        atlas_t2,
        atlas_mask,
        data_root=args.atlas_bank_data_root,
        image_size=image_size,
        mask_threshold=args.mask_threshold,
        max_cases=args.max_cases,
        skip_case_id=atlas_case_id if args.skip_atlas_case else '',
    )
    if len(dataset) == 0:
        raise RuntimeError('No training pairs found in {}'.format(split_file))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = LightRegistrationNet(args.transform, args.grid_size, args.max_disp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / 'light_register_config.json').open('w') as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            fixed = batch['fixed'].to(device)
            moving = batch['moving'].to(device)
            fixed_mask = batch['fixed_mask'].to(device)
            moving_mask = batch['moving_mask'].to(device)

            warped, theta, ddf = model(moving, fixed)
            loss_img = ncc_loss(warped, fixed)
            loss = args.image_loss_weight * loss_img

            loss_mask = torch.tensor(0.0, device=device)
            if args.supervision == 'mask':
                warped_mask = warp_moving_mask(model, moving_mask, fixed, theta, ddf)
                if fixed_mask.sum() > 0 and moving_mask.sum() > 0:
                    loss_mask = dice_loss(warped_mask, fixed_mask)
                    loss = loss + args.mask_loss_weight * loss_mask

            loss_smooth = torch.tensor(0.0, device=device)
            if ddf is not None:
                loss_smooth = ddf_smoothness(ddf)
                loss = loss + args.smooth_weight * loss_smooth

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print('epoch {}, loss={:.5f}'.format(epoch, mean_loss))
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, save_dir / 'latest_light_register.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, save_dir / 'epoch_{}_light_register.pth'.format(epoch))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a lightweight atlas-to-target registration network')
    parser.add_argument('--data_root', type=str, default='../../Datasets/ProstateDatasets/data_split_files', help='Path to split-file root, same as segment.py')
    parser.add_argument('--path', type=str, default='4-prostate158', help='Dataset split subdirectory, same as segment.py')
    parser.add_argument('--phase', type=str, choices=['train', 'val', 'test'], default='train', help='Split name to train from')
    parser.add_argument('--atlas_bank_dir', type=str, default='../register/atlas_bank', help='Directory containing reference_bank.csv')
    parser.add_argument('--atlas_bank_data_root', type=str, default=DEFAULT_ATLAS_BANK_DATA_ROOT, help='Readable dataset root used to remap stale atlas-bank and split-file paths')
    parser.add_argument('--atlas_case_id', type=str, default='', help='Optional atlas case id from reference_bank.csv; default uses first readable atlas')
    parser.add_argument('--skip_atlas_case', action='store_true', help='Skip the selected atlas case if it appears in the target split')
    parser.add_argument('--save_dir', default='checkpoints/light_register', help='Output checkpoint directory')
    parser.add_argument('--transform', choices=['affine', 'ddf'], default='ddf', help='Transform head to train')
    parser.add_argument('--grid_size', type=int, nargs=3, default=[10, 10, 10], help='Sparse DDF grid size for transform=ddf')
    parser.add_argument('--max_disp', type=float, default=0.20, help='Max normalized displacement for DDF, where 2.0 spans the full image axis')
    parser.add_argument('--supervision', choices=['none', 'mask'], default='none', help='Use target mask supervision in addition to image similarity')
    parser.add_argument('--mask_threshold', type=float, default=1.0, help='Foreground mask is label > this threshold, e.g. 1 for label>1')
    parser.add_argument('--image_size', type=int, nargs=3, default=[96, 96, 96], help='Training resize shape D H W; omit with no value is not supported')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--image_loss_weight', type=float, default=1.0)
    parser.add_argument('--mask_loss_weight', type=float, default=1.0)
    parser.add_argument('--smooth_weight', type=float, default=0.05)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--max_cases', type=int, default=0, help='Limit cases for debugging; 0 uses all cases')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    return args


if __name__ == '__main__':
    train(parse_args())
