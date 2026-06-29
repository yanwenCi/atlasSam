import torch
import torch.nn as nn
import numpy as np
import os
import json
import csv
from os.path import join
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from dataloader3d import dataset_loaders
loss_fn = None
eval_metrics = None
nets = None
import argparse
import logging
roc_curve = precision_recall_curve = roc_auc_score = None
plt = None
import cv2
import json
import nibabel as nib
import torch.nn.functional as F
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTER_DIR = ROOT_DIR / 'register'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(REGISTER_DIR) not in sys.path:
    sys.path.insert(0, str(REGISTER_DIR))
from light_register import LightRegistrationNet, normalize_image, warp_mask_with_ddf

DEFAULT_ATLAS_BANK_DATA_ROOT = (
    '/cluster/project7/longitude/Datasets/ProstateDatasets/3-picai-data/data-ROI-192-96'
)
LEGACY_ATLAS_BANK_ROOTS = (
    '/raid/candi/Wen/Dataset/ProstateDatasets/3-picai-data/data-ROI-192-96',
    '/raid/candi/Wen/ProstateSeg/Data/scripts/3-picai-data/data-ROI-192-96',
)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sam_slice_image(slice_2d):
    """Convert one normalized 2D medical slice to SAM's uint8 RGB input."""
    image = np.asarray(slice_2d, dtype=np.float32)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=-1)


def mask_to_box(mask, pad=3):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w - 1)
    y1 = min(int(ys.max()) + pad + 1, h - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def mask_to_points(mask, num_points=1):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    if num_points <= 1:
        cy = int(np.round(ys.mean()))
        cx = int(np.round(xs.mean()))
        coords = np.array([[cx, cy]], dtype=np.float32)
    else:
        idx = np.linspace(0, len(xs) - 1, min(num_points, len(xs))).astype(int)
        coords = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    labels = np.ones((coords.shape[0],), dtype=np.int32)
    return coords, labels


def dice_binary(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    return (2.0 * np.logical_and(pred, target).sum() + eps) / (pred.sum() + target.sum() + eps)


def hd95_binary(pred, target, spacing=(1, 1, 1), empty_value=100.0):
    pred_points = np.argwhere(np.asarray(pred) > 0)
    target_points = np.argwhere(np.asarray(target) > 0)
    if pred_points.size == 0 or target_points.size == 0:
        return float(empty_value)
    from scipy.spatial import KDTree
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


def build_prompt_from_mask(mask, prompt_type='box', box_pad=3, num_points=1):
    box = None
    point_coords, point_labels = None, None
    if prompt_type in ('box', 'box_points'):
        box = mask_to_box(mask, pad=box_pad)
    if prompt_type in ('points', 'box_points'):
        point_coords, point_labels = mask_to_points(mask, num_points=num_points)
    return box, point_coords, point_labels


def predict_volume_with_sam(predictor, image_vol, prompt_mask, args):
    """Run slice-wise promptable SAM/MedSAM inference and rebuild a 3D mask."""
    pred = np.zeros_like(prompt_mask, dtype=np.uint8)
    num_slices = image_vol.shape[-1]
    for z in range(num_slices):
        mask_slice = prompt_mask[..., z] > 0
        if mask_slice.sum() < args.min_prompt_pixels:
            continue

        box, point_coords, point_labels = build_prompt_from_mask(
            mask_slice,
            prompt_type=args.prompt_type,
            box_pad=args.box_pad,
            num_points=args.num_prompt_points,
        )
        if box is None and point_coords is None:
            continue

        predictor.set_image(sam_slice_image(image_vol[..., z]))
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=args.multimask_output,
        )
        best = int(np.argmax(scores)) if args.multimask_output else 0
        pred[..., z] = masks[best].astype(np.uint8)
    return pred



def resize_volume_np(volume, shape, mode='nearest'):
    """Resize a DHW numpy volume with torch interpolation."""
    if tuple(volume.shape) == tuple(shape):
        return volume
    tensor = torch.from_numpy(volume.astype(np.float32))[None, None]
    interp_mode = 'nearest' if mode == 'nearest' else 'trilinear'
    kwargs = {} if interp_mode == 'nearest' else {'align_corners': False}
    resized = F.interpolate(tensor, size=tuple(shape), mode=interp_mode, **kwargs)
    return resized[0, 0].cpu().numpy()



def dilate_mask_np(mask, radius):
    if radius <= 0:
        return mask
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool3d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return dilated[0, 0].numpy() > 0


def crop_or_pad_3d(volume, center, crop_size, pad_value=0):
    """Return a fixed-size DHW crop and slices for copying crop results back."""
    shape = np.array(volume.shape)
    size = np.array([crop_size] * 3 if np.isscalar(crop_size) else crop_size, dtype=int)
    center = np.array(center, dtype=int)
    start = center - size // 2
    end = start + size

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, shape)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    crop = np.full(tuple(size), pad_value, dtype=volume.dtype)
    src = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_end))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_end))
    crop[dst] = volume[src]
    return crop, src, dst


def normalize_sammed3d_roi(image):
    image = image.astype(np.float32)
    mask = image > 0
    voxels = image[mask] if mask.any() else image.reshape(-1)
    mean = float(voxels.mean())
    std = float(voxels.std())
    if std < 1e-6:
        std = 1.0
    return (image - mean) / std


def prompt_points_from_prior(prior_mask, num_points=1):
    coords = np.argwhere(prior_mask > 0)
    if coords.size == 0:
        return torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.long)
    centroid = coords.mean(axis=0, keepdims=True)
    distances = ((coords - centroid) ** 2).sum(axis=1)
    first = coords[int(np.argmin(distances))]
    if num_points <= 1:
        points = first[None]
    else:
        order = np.linspace(0, len(coords) - 1, min(num_points, len(coords))).astype(int)
        points = np.vstack([first[None], coords[order]])[:num_points]
    return (
        torch.from_numpy(points.astype(np.float32))[None],
        torch.ones(1, points.shape[0], dtype=torch.long),
    )



def zones_to_atlas_mask(zones, cg_label=2, pz_label=1):
    """Convert source prostate_zones labels to atlas convention: PZ=1, CG=2."""
    atlas_mask = np.zeros_like(zones, dtype=np.uint8)
    atlas_mask[zones == pz_label] = 1
    atlas_mask[zones == cg_label] = 2
    return atlas_mask


def load_single_atlas_record(args):
    atlas_npz = np.load(args.atlas_path)
    if 'mask_atlas' not in atlas_npz:
        raise KeyError('{} must contain a `mask_atlas` array with PZ=1 and CG=2 labels'.format(args.atlas_path))
    record = {'case_id': Path(args.atlas_path).stem, 'mask': atlas_npz['mask_atlas'].astype(np.uint8)}
    if 'atlas' in atlas_npz:
        record['t2'] = atlas_npz['atlas'].astype(np.float32)
    return record


def atlas_bank_data_root(args, bank_dir):
    metadata_path = Path(bank_dir) / 'bank_metadata.json'
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            data_root = metadata.get('data_root', '')
            if data_root and Path(data_root).exists():
                return data_root
        except Exception as exc:
            print('Warning: could not read atlas-bank metadata: {}'.format(exc))
    return getattr(args, 'atlas_bank_data_root', '') or DEFAULT_ATLAS_BANK_DATA_ROOT


def resolve_bank_path(bank_dir, value, data_root=None):
    if value is None or value == '' or value == 'None':
        return None
    path = Path(value)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = Path(bank_dir) / path
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
    if data_root and case_parts:
        candidate = Path(data_root) / case_parts[-1] / path.name
        if candidate.exists():
            return candidate
    return path


def load_atlas_bank_masks(args):
    """Load selected atlas masks from reference_bank.csv if an atlas bank is present."""
    if not args.atlas_bank_dir:
        return []
    bank_dir = Path(args.atlas_bank_dir)
    reference_csv = bank_dir / 'reference_bank.csv'
    if not reference_csv.exists():
        return []

    data_root = atlas_bank_data_root(args, bank_dir)
    atlas_masks = []
    seen_cases = set()
    with open(reference_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get('case_id', 'atlas_{}'.format(len(atlas_masks)))
            if case_id in seen_cases:
                continue
            zones_path = (
                row.get('bank_zones_path')
                or row.get('zones_path')
                or row.get('prostate_zones_path')
                or row.get('bank_cg_mask_path')
                or row.get('cg_mask_path')
            )
            zones_path = resolve_bank_path(bank_dir, zones_path, data_root=data_root)
            if zones_path is None or not zones_path.exists():
                print("Warning: skipping atlas-bank row without readable zones_path: {}".format(row.get('case_id', '<unknown>')))
                continue

            if zones_path.name.endswith('_cg.nii.gz') or row.get('bank_cg_mask_path') or row.get('cg_mask_path'):
                cg_path = zones_path
                pz_path = resolve_bank_path(bank_dir, row.get('bank_pz_mask_path') or row.get('pz_mask_path'), data_root=data_root)
                if pz_path is None or not pz_path.exists():
                    print("Warning: skipping atlas-bank row without readable pz mask: {}".format(row.get('case_id', '<unknown>')))
                    continue
                cg = nib.load(str(cg_path)).get_fdata() > 0
                pz = nib.load(str(pz_path)).get_fdata() > 0
                atlas_mask = np.zeros(cg.shape, dtype=np.uint8)
                atlas_mask[cg] = 1
                atlas_mask[pz] = 2
            else:
                zones = nib.load(str(zones_path)).get_fdata()
                atlas_mask = zones_to_atlas_mask(zones, cg_label=args.cg_label, pz_label=args.pz_label)

            if np.any(atlas_mask == 1) and np.any(atlas_mask == 2):
                t2_path = resolve_bank_path(bank_dir, row.get('bank_t2_path') or row.get('t2_path'), data_root=data_root)
                feature_values = {}
                for key_name, value in row.items():
                    try:
                        feature_values[key_name] = float(value)
                    except (TypeError, ValueError):
                        pass
                atlas_masks.append({
                    'case_id': case_id,
                    'mask': atlas_mask,
                    't2_path': str(t2_path) if t2_path is not None else '',
                    'zones_path': str(zones_path),
                    'features': feature_values,
                })
                seen_cases.add(case_id)
            else:
                print("Warning: skipping atlas-bank row with empty CG/PZ mask: {}".format(case_id))

    if atlas_masks:
        print("Loaded {} selected atlas masks from {}".format(len(atlas_masks), reference_csv))
    return atlas_masks


def load_conditioning_atlases(args):
    bank_masks = load_atlas_bank_masks(args)
    if bank_masks:
        return bank_masks
    return [load_single_atlas_record(args)]


def setup_light_register(checkpoint_path, device):
    """Load the frozen lightweight atlas-to-image registration model."""
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get('args', {})
    model = LightRegistrationNet(
        transform=ckpt_args.get('transform', 'ddf'),
        grid_size=ckpt_args.get('grid_size', [10, 10, 10]),
        max_disp=ckpt_args.get('max_disp', 0.20),
    ).to(device)
    model.load_state_dict(checkpoint.get('model', checkpoint))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print('Loaded frozen light registration from {}'.format(checkpoint_path))
    return model


def _atlas_t2_from_record(atlas_record):
    if 't2' in atlas_record:
        return np.asarray(atlas_record['t2'], dtype=np.float32)
    t2_path = atlas_record.get('t2_path', '')
    if t2_path and os.path.exists(t2_path):
        return nib.load(t2_path).get_fdata().astype(np.float32)
    return None


def _resize_tensor_to_shape(tensor, shape, mode):
    if tuple(tensor.shape[-3:]) == tuple(shape):
        return tensor
    kwargs = {} if mode == 'nearest' else {'align_corners': False}
    return F.interpolate(tensor, size=tuple(shape), mode=mode, **kwargs)


def _warp_mask_with_light_register(light_model, mask_tensor, fixed_tensor, theta, ddf):
    if light_model.transform == 'affine':
        grid = F.affine_grid(theta, fixed_tensor.shape, align_corners=True)
        return F.grid_sample(mask_tensor, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    return warp_mask_with_ddf(mask_tensor, ddf)


def register_atlas_mask_to_image(atlas_record, image, args, light_model=None, device='cpu'):
    """Return atlas mask in current image space, using frozen light registration when available."""
    atlas_mask = np.asarray(atlas_record['mask'], dtype=np.uint8)
    if light_model is None:
        return resize_volume_np(atlas_mask, image.shape, mode='nearest').astype(np.uint8)

    atlas_t2 = _atlas_t2_from_record(atlas_record)
    if atlas_t2 is None:
        print('Warning: atlas {} has no T2 image; using resized mask without light registration'.format(atlas_record.get('case_id', '')))
        return resize_volume_np(atlas_mask, image.shape, mode='nearest').astype(np.uint8)

    fixed = torch.from_numpy(normalize_image(image).astype(np.float32))[None, None].to(device)
    moving = torch.from_numpy(normalize_image(atlas_t2).astype(np.float32))[None, None].to(device)
    cg = torch.from_numpy((atlas_mask == 2).astype(np.float32))[None, None].to(device)
    pz = torch.from_numpy((atlas_mask == 1).astype(np.float32))[None, None].to(device)

    moving = _resize_tensor_to_shape(moving, image.shape, 'trilinear')
    cg = _resize_tensor_to_shape(cg, image.shape, 'nearest')
    pz = _resize_tensor_to_shape(pz, image.shape, 'nearest')

    with torch.no_grad():
        _, theta, ddf = light_model(moving, fixed)
        cg_warp = _warp_mask_with_light_register(light_model, cg, fixed, theta, ddf)
        pz_warp = _warp_mask_with_light_register(light_model, pz, fixed, theta, ddf)

    cg_np = cg_warp[0, 0].cpu().numpy() > 0.5
    pz_np = pz_warp[0, 0].cpu().numpy() > 0.5
    registered = np.zeros(image.shape, dtype=np.uint8)
    registered[pz_np] = 1
    registered[cg_np] = 2
    return registered


def choose_conditioning_atlas(atlases, case_key, index, mode='cycle'):
    if not atlases:
        raise RuntimeError('No conditioning atlas masks are available')
    if mode == 'first':
        return atlases[0]
    if mode == 'random':
        return atlases[np.random.randint(0, len(atlases))]
    # Stable cycle keeps runs reproducible and avoids using labels for atlas selection.
    return atlases[index % len(atlases)]


def _batch_string(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return str(value)


def _feature_names(args):
    return [name.strip() for name in args.atlas_retrieval_features.split(',') if name.strip()]


def target_retrieval_features(data, image, args):
    features = {}
    t2_path = data.get('t2_path', '')
    t2_path = _batch_string(t2_path) if t2_path else ''
    voxels = None
    if t2_path and os.path.exists(t2_path):
        try:
            raw = nib.load(t2_path).get_fdata().astype(np.float32)
            voxels = raw[raw > 0]
            if voxels.size == 0:
                voxels = raw.reshape(-1)
        except Exception as exc:
            print('Warning: could not read target T2 for atlas retrieval: {}'.format(exc))
    if voxels is None:
        raw = np.asarray(image, dtype=np.float32)
        voxels = raw[raw > 0]
        if voxels.size == 0:
            voxels = raw.reshape(-1)
    features['whole_t2_mean'] = float(voxels.mean())
    features['whole_t2_std'] = float(voxels.std())
    features['whole_t2_median'] = float(np.median(voxels))
    return features


def atlas_feature_stats(atlases, feature_names):
    stats = {}
    for name in feature_names:
        vals = [atlas.get('features', {}).get(name) for atlas in atlases]
        vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float32)
        if vals.size:
            std = float(vals.std())
            stats[name] = (float(vals.mean()), std if std > 1e-6 else 1.0)
    return stats


def retrieve_topk_atlases(atlases, target_features, args):
    if not atlases:
        return []
    if args.atlas_top_k <= 0:
        return atlases
    feature_names = _feature_names(args)
    stats = atlas_feature_stats(atlases, feature_names)
    scored = []
    for atlas in atlases:
        dist_terms = []
        for name in feature_names:
            if name not in target_features or name not in stats:
                continue
            atlas_value = atlas.get('features', {}).get(name)
            if atlas_value is None or not np.isfinite(atlas_value):
                continue
            _, std = stats[name]
            dist_terms.append(((float(atlas_value) - float(target_features[name])) / std) ** 2)
        if dist_terms:
            distance = float(np.sqrt(np.mean(dist_terms)))
            similarity = float(np.exp(-distance / max(args.atlas_similarity_temperature, 1e-6)))
        else:
            distance = 0.0
            similarity = 1.0
        item = dict(atlas)
        item['retrieval_distance'] = distance
        item['similarity_weight'] = similarity
        scored.append(item)
    scored.sort(key=lambda x: x['retrieval_distance'])
    return scored[:min(args.atlas_top_k, len(scored))]


def prompt_consistency_weight(pred_prob, prior_mask, threshold):
    pred_mask = pred_prob > threshold
    if pred_mask.sum() == 0 or prior_mask.sum() == 0:
        return 0.0
    overlap = dice_binary(pred_mask, prior_mask > 0)
    confidence = float(pred_prob[pred_mask].mean()) if pred_mask.any() else 0.0
    return max(1e-6, 0.5 * float(overlap) + 0.5 * confidence)


def normalize_atlas_weights(weighted_preds):
    total = sum(item['weight'] for item in weighted_preds)
    if total <= 0:
        fallback = 1.0 / max(1, len(weighted_preds))
        for item in weighted_preds:
            item['weight'] = fallback
        return weighted_preds
    for item in weighted_preds:
        item['weight'] = item['weight'] / total
    return weighted_preds


def fuse_weighted_atlas_predictions(weighted_preds, label_shape, threshold):
    cg_score = np.zeros(label_shape, dtype=np.float32)
    pz_score = np.zeros(label_shape, dtype=np.float32)
    for item in weighted_preds:
        pred = item['pred']
        weight = item['weight']
        cg_score += weight * (pred == 2).astype(np.float32)
        pz_score += weight * (pred == 1).astype(np.float32)
    pred = np.zeros(label_shape, dtype=np.uint8)
    pred[(pz_score >= threshold) & (pz_score > cg_score)] = 1
    pred[(cg_score >= threshold) & (cg_score >= pz_score)] = 2
    return pred, cg_score, pz_score


def load_sammed3d_model(args, device):
    """Load SAM-Med3D using the official MedIM entry point from uni-medical/SAM-Med3D."""
    try:
        import medim
    except ModuleNotFoundError as exc:
        if exc.name == 'medim':
            raise ImportError(
                'model_type=sammed3d_atlas requires the SAM-Med3D MedIM dependency. '
                'Install it with `pip install medim` in the active environment.'
            ) from exc
        raise ImportError(
            'MedIM is installed, but it is missing dependency `{}`. '
            'Install it in the active environment, for example `pip install {}`.'.format(exc.name, exc.name)
        ) from exc

    checkpoint = args.sam_checkpoint or 'https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth'
    model = medim.create_model('SAM-Med3D', pretrained=True, checkpoint_path=checkpoint)
    model = model.to(device)
    model.eval()
    return model



class SamMed3DBottleneckAdapter(nn.Module):
    """Small residual bottleneck trained on frozen SAM-Med3D image embeddings."""
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.down = None
        self.act = nn.GELU()
        self.up = None

    def build(self, in_channels, device, dtype):
        if self.down is not None:
            return
        hidden = max(1, min(self.hidden_channels, int(in_channels)))
        self.down = nn.Conv3d(int(in_channels), hidden, kernel_size=1).to(device=device, dtype=dtype)
        self.up = nn.Conv3d(hidden, int(in_channels), kernel_size=1).to(device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        self.build(x.shape[1], x.device, x.dtype)
        return x + self.up(self.act(self.down(x)))


def freeze_sammed3d_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def initialize_sammed3d_bottleneck(model, args, device):
    bottleneck = SamMed3DBottleneckAdapter(args.medsam_bottleneck_channels).to(device)
    side = int(args.sam_crop_size) if np.isscalar(args.sam_crop_size) else int(args.sam_crop_size[0])
    dummy = torch.zeros(1, 1, side, side, side, device=device)
    with torch.no_grad():
        embeddings = model.image_encoder(dummy)
    bottleneck(embeddings)
    return bottleneck


def binary_dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean()


def sammed3d_forward_logits(model, image_roi, prior_roi, args, device, bottleneck=None):
    image_roi = normalize_sammed3d_roi(image_roi)
    image_tensor = torch.from_numpy(image_roi.astype(np.float32))[None, None].to(device)
    prior_tensor = torch.from_numpy((prior_roi > 0).astype(np.float32))[None, None].to(device)

    low_res_size = tuple(max(1, s // 4) for s in image_roi.shape)
    low_res_mask = F.interpolate(prior_tensor, size=low_res_size, mode='trilinear', align_corners=False)

    with torch.no_grad():
        image_embeddings = model.image_encoder(image_tensor)
    if bottleneck is not None:
        image_embeddings = bottleneck(image_embeddings)

    point_coords, point_labels = prompt_points_from_prior(prior_roi, args.num_prompt_points)
    point_coords = point_coords.to(device)
    point_labels = point_labels.to(device)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[point_coords, point_labels],
            boxes=None,
            masks=low_res_mask,
        )
        image_pe = model.prompt_encoder.get_dense_pe()
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
    )
    return F.interpolate(low_res_masks, size=image_roi.shape, mode='trilinear', align_corners=False)


def box_mask_3d(mask, pad=0):
    coords = np.argwhere(mask > 0)
    boxed = np.zeros_like(mask, dtype=np.uint8)
    if coords.size == 0:
        return boxed
    lo = np.maximum(coords.min(axis=0) - int(pad), 0)
    hi = np.minimum(coords.max(axis=0) + int(pad) + 1, np.asarray(mask.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    boxed[slices] = 1
    return boxed


def atlas_conditioned_medsam_rois(image, label, atlas_mask, args):
    """Yield CG and whole-prostate binary tasks; PZ is derived as prostate minus CG."""
    atlas_for_image = resize_volume_np(atlas_mask, image.shape, mode='nearest').astype(np.uint8)
    tasks = (
        ('CG', atlas_for_image == 2, label == 2),
        ('Prostate', atlas_for_image > 0, label > 0),
    )
    for name, prior, target in tasks:
        if prior.sum() < args.min_prompt_pixels and target.sum() < args.min_prompt_pixels:
            continue
        anchor = prior if prior.sum() >= args.min_prompt_pixels else target
        center = np.round(np.argwhere(anchor > 0).mean(axis=0)).astype(int)
        prompt_prior = box_mask_3d(prior, pad=args.atlas_prompt_pad)
        if prompt_prior.sum() < args.min_prompt_pixels:
            prompt_prior = box_mask_3d(target, pad=args.atlas_prompt_pad)
        image_roi, src, dst = crop_or_pad_3d(image, center, args.sam_crop_size, pad_value=0)
        prior_roi, _, _ = crop_or_pad_3d(prompt_prior.astype(np.uint8), center, args.sam_crop_size, pad_value=0)
        target_roi, _, _ = crop_or_pad_3d(target.astype(np.uint8), center, args.sam_crop_size, pad_value=0)
        yield name, image_roi, prior_roi, target_roi, src, dst


def load_sammed3d_finetune_checkpoint(model, bottleneck, checkpoint, device):
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    bottleneck_state = None
    if isinstance(state, dict) and 'bottleneck' in state:
        bottleneck_state = state['bottleneck']
    elif isinstance(state, dict) and any(key.startswith(('down.', 'up.')) for key in state):
        bottleneck_state = state
    if bottleneck is None or bottleneck_state is None:
        print('Warning: {} does not contain bottleneck weights; keeping frozen SAM-Med3D unchanged'.format(checkpoint))
        return
    bottleneck.load_state_dict(bottleneck_state, strict=False)


def atlas_conditioned_sammed3d_train(args):
    """Fine-tune SAM-Med3D with atlas-conditioned CG/PZ prompts."""
    device = setup_device(args.gpus)
    model = load_sammed3d_model(args, device)
    freeze_sammed3d_model(model)
    bottleneck = initialize_sammed3d_bottleneck(model, args, device)
    save_dir = join(args.save_dir, '{}_{}'.format(args.model_type, args.path))
    resume_checkpoint = args.sam_finetune_checkpoint
    if args.continue_train and not resume_checkpoint:
        candidate = join(save_dir, 'latest_sammed3d_atlas.pth')
        resume_checkpoint = candidate if os.path.exists(candidate) else ''
    load_sammed3d_finetune_checkpoint(model, bottleneck, resume_checkpoint, device)

    trainable = [p for p in bottleneck.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('No bottleneck parameters were selected for fine-tuning')
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    conditioning_atlases = load_conditioning_atlases(args)
    light_register = setup_light_register(args.pretrained_light_register, device)

    data_path = join(args.data_root, args.path)
    train_loader = get_dataloader(data_path, 'train', 1, args.crop_size, istest=False, spacing=args.spacing)
    os.makedirs(save_dir, exist_ok=True)
    save_config(args, save_dir)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.eval()
        bottleneck.train()
        epoch_losses = []

        for i, data in enumerate(train_loader):
            image = _to_numpy(data['img'])[0, 0].astype(np.float32)
            label = _to_numpy(data['seg'])[0].astype(np.uint8)
            losses = []

            key = data['key'][0] if isinstance(data['key'], (list, tuple)) else str(data['key'])
            atlas_record = choose_conditioning_atlas(conditioning_atlases, key, i, mode=args.atlas_bank_mode)
            atlas_mask = register_atlas_mask_to_image(atlas_record, image, args, light_register, device)
            for name, image_roi, prior_roi, target_roi, _, _ in atlas_conditioned_medsam_rois(image, label, atlas_mask, args):
                target = torch.from_numpy(target_roi.astype(np.float32))[None, None].to(device)
                logits = sammed3d_forward_logits(
                    model,
                    image_roi,
                    prior_roi,
                    args,
                    device,
                    bottleneck=bottleneck,
                )
                bce = F.binary_cross_entropy_with_logits(logits, target)
                dice = binary_dice_loss_from_logits(logits, target)
                losses.append(args.bce_loss_weight * bce + args.dice_loss_weight * dice)

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

            if i % args.print_freq == 0:
                print("Epoch {}, batch {}, {}, atlas={}, atlas-conditioned SAM-Med3D loss={:.4f}".format(epoch, i, key, atlas_record['case_id'], epoch_losses[-1]))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('inf')
        print('Epoch {}, mean atlas-conditioned SAM-Med3D loss={:.4f}'.format(epoch, mean_loss))

        latest_path = join(save_dir, 'latest_sammed3d_atlas.pth')
        state = {'bottleneck': bottleneck.state_dict(), 'epoch': epoch, 'loss': mean_loss, 'args': vars(args)}
        torch.save(state, latest_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(state, join(save_dir, 'best_sammed3d_atlas.pth'))
        if (epoch + 1) % args.save_freq == 0:
            torch.save(state, join(save_dir, 'epoch_{}_sammed3d_atlas.pth'.format(epoch)))

    return best_loss


@torch.no_grad()
def sammed3d_infer_from_atlas_prior(model, bottleneck, image_roi, prior_roi, args, device):
    """Run one binary SAM-Med3D segmentation conditioned on an atlas prior ROI."""
    image_roi = normalize_sammed3d_roi(image_roi)
    image_tensor = torch.from_numpy(image_roi.astype(np.float32))[None, None].to(device)
    prior_tensor = torch.from_numpy((prior_roi > 0).astype(np.float32))[None, None].to(device)

    low_res_size = tuple(max(1, s // 4) for s in image_roi.shape)
    low_res_mask = F.interpolate(prior_tensor, size=low_res_size, mode='trilinear', align_corners=False)

    image_embeddings = model.image_encoder(image_tensor)
    if bottleneck is not None:
        image_embeddings = bottleneck(image_embeddings)
    point_coords, point_labels = prompt_points_from_prior(prior_roi, args.num_prompt_points)
    point_coords = point_coords.to(device)
    point_labels = point_labels.to(device)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=[point_coords, point_labels],
        boxes=None,
        masks=low_res_mask,
    )
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
    )
    masks = F.interpolate(low_res_masks, size=image_roi.shape, mode='trilinear', align_corners=False)
    prob = torch.sigmoid(masks)[0, 0].cpu().numpy()
    return prob.astype(np.float32)


def atlas_conditioned_sammed3d_test(args):
    """Segment CG/PZ with 3D SAM-Med3D using atlas masks as prompts/priors."""
    device = setup_device(args.gpus)
    model = load_sammed3d_model(args, device)
    freeze_sammed3d_model(model)
    bottleneck = initialize_sammed3d_bottleneck(model, args, device)
    checkpoint = args.sam_finetune_checkpoint
    if not checkpoint and args.load_from_dir:
        candidate = join(args.save_dir, args.load_from_dir, 'best_sammed3d_atlas.pth')
        checkpoint = candidate if os.path.exists(candidate) else ''
    print('Loading SAM-Med3D bottleneck from {}'.format(checkpoint))
    load_sammed3d_finetune_checkpoint(model, bottleneck, checkpoint, device)
    model.eval()
    bottleneck.eval()

    conditioning_atlases = load_conditioning_atlases(args)
    light_register = setup_light_register(args.pretrained_light_register, device)

    data_path = join(args.data_root, args.path)
    loader = get_dataloader(data_path, 'test', 1, args.crop_size, istest=True, spacing=args.spacing)
    save_dir = join(args.save_dir, args.load_from_dir or '{}_{}'.format(args.model_type, args.path))
    visual_dir = join(save_dir, '{}_sammed3d_atlas'.format(args.path))
    os.makedirs(visual_dir, exist_ok=True)
    print(visual_dir)
    metrics = {}
    rows = []
    dist_rows = []
    for i, data in enumerate(loader):
        image = _to_numpy(data['img'])[0, 0].astype(np.float32)
        label = _to_numpy(data['seg'])[0].astype(np.uint8)
        # print(np.unique(label, return_counts=True))
        key = data['key'][0] if isinstance(data['key'], (list, tuple)) else str(data['key'])
        if args.atlas_bank_mode == 'topk':
            target_features = target_retrieval_features(data, image, args)
            atlas_records = retrieve_topk_atlases(conditioning_atlases, target_features, args)
        elif args.atlas_bank_mode == 'ensemble':
            atlas_records = conditioning_atlases
        else:
            atlas_records = [choose_conditioning_atlas(conditioning_atlases, key, i, mode=args.atlas_bank_mode)]
        atlas_for_image = None
        weighted_preds = []

        for atlas_record in atlas_records:
            # atlas_for_image = register_atlas_mask_to_image(atlas_record, image, args, light_register, device)
            task_probs = {}
            atlas_reliability = []
            atlas_for_image = label
            for name, prior in (('CG', atlas_for_image == 2), ('Prostate', atlas_for_image > 0)):
                if prior.sum() < args.min_prompt_pixels:
                    continue
                prompt_prior = box_mask_3d(prior, pad=args.atlas_prompt_pad)
                center = np.round(np.argwhere(prior > 0).mean(axis=0)).astype(int)
                image_roi, src, dst = crop_or_pad_3d(image, center, args.sam_crop_size, pad_value=0)
                prior_roi, _, _ = crop_or_pad_3d(prompt_prior.astype(np.uint8), center, args.sam_crop_size, pad_value=0)
                if prior_roi.sum() < args.min_prompt_pixels:
                    continue
                cls_prob_roi = sammed3d_infer_from_atlas_prior(model, bottleneck, image_roi, prior_roi, args, device)
                cls_prob = np.zeros_like(label, dtype=np.float32)
                cls_prob[src] = cls_prob_roi[dst]
                task_probs[name] = cls_prob
                atlas_reliability.append(prompt_consistency_weight(cls_prob, prompt_prior, args.sam_prob_threshold))

            atlas_pred = np.zeros_like(label, dtype=np.uint8)
            cg_pred = task_probs.get('CG', np.zeros_like(label, dtype=np.float32)) > args.sam_prob_threshold
            prostate_pred = task_probs.get('Prostate', np.zeros_like(label, dtype=np.float32)) > args.sam_prob_threshold
            atlas_pred[prostate_pred & ~cg_pred] = 1
            atlas_pred[cg_pred] = 2
            if np.any(atlas_pred > 0):
                prompt_weight = float(np.mean(atlas_reliability)) if atlas_reliability else 1.0
                similarity_weight = float(atlas_record.get('similarity_weight', 1.0))
                weighted_preds.append({
                    'case_id': atlas_record.get('case_id', ''),
                    'pred': atlas_pred,
                    'weight': similarity_weight * prompt_weight,
                    'similarity_weight': similarity_weight,
                    'prompt_weight': prompt_weight,
                })

        if args.atlas_bank_mode in ('topk', 'ensemble') and len(weighted_preds) > 1:
            weighted_preds = normalize_atlas_weights(weighted_preds)
        
            pred, cg_score, pz_score = fuse_weighted_atlas_predictions(weighted_preds, label.shape, args.atlas_fusion_threshold)
        else:
            pred = weighted_preds[0]['pred'] if weighted_preds else np.zeros_like(label, dtype=np.uint8)
            if weighted_preds:
                weighted_preds[0]['weight'] = 1.0

        if args.atlas_bank_mode == 'topk':
            selected = ', '.join(['{}:{:.3f}'.format(item['case_id'], item['weight']) for item in weighted_preds])
            print('{}: {}, top-k atlas weights [{}]'.format(i, key, selected))

        cg_dice = float(dice_binary(pred == 2, label == 2))
        pz_dice = float(dice_binary(pred == 1, label == 1))
        prostate_dice = float(dice_binary(pred > 0, label > 0))
        cg_dist = hd95_binary(pred == 2, label == 2, spacing=args.spacing)
        pz_dist = hd95_binary(pred == 1, label == 1, spacing=args.spacing)
        prostate_dist = hd95_binary(pred > 0, label > 0, spacing=args.spacing)
        metrics[key] = {
            'CG Dice': cg_dice,
            'PZ Dice': pz_dice,
            'WG Dice': prostate_dice,
            'Prostate Dice': prostate_dice,
            'CG Dist': cg_dist,
            'PZ Dist': pz_dist,
            'WG Dist': prostate_dist,
            'Prostate Dist': prostate_dist,
            'Atlas Weights': [
                {
                    'case_id': item.get('case_id', ''),
                    'weight': float(item.get('weight', 0.0)),
                    'similarity_weight': float(item.get('similarity_weight', 1.0)),
                    'prompt_weight': float(item.get('prompt_weight', 1.0)),
                }
                for item in weighted_preds
            ],
        }
        rows.append([cg_dice, pz_dice, prostate_dice])
        dist_rows.append([cg_dist, pz_dist, prostate_dist])
        print('{}: {}, CG Dice={:.4f}, PZ Dice={:.4f}, prostate Dice={:.4f}, CG Dist={:.4f}, PZ Dist={:.4f}, prostate Dist={:.4f}'.format(
            i, key, cg_dice, pz_dice, prostate_dice, cg_dist, pz_dist, prostate_dist
        ))

        if args.save_output:
            affine = _to_numpy(data['affine'])[0] if torch.is_tensor(data['affine']) else data['affine'][0]
            nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), join(visual_dir, '{}_pred.nii.gz'.format(key)))
            mask_for_slice = (label > 0) | (pred > 0)
            z = int(np.median(np.where(mask_for_slice)[-1])) if mask_for_slice.sum() > 0 else image.shape[-1] // 2
            concat = np.concatenate([
                255 * np.rot90(image[..., z], k=3),
                127 * np.rot90(pred[..., z].astype(np.uint8), k=3),
                127 * np.rot90(label[..., z].astype(np.uint8), k=3),
                127 * np.rot90(atlas_for_image[..., z].astype(np.uint8), k=3),
            ], axis=1)

            cv2.imwrite(join(visual_dir, '{}_z{}.png'.format(key, z)), concat)

    rows = np.asarray(rows, dtype=np.float32)
    dist_rows = np.asarray(dist_rows, dtype=np.float32)
    mean = rows.mean(axis=0) if len(rows) else np.zeros(3, dtype=np.float32)
    std = rows.std(axis=0) if len(rows) else np.zeros(3, dtype=np.float32)
    dist_mean = dist_rows.mean(axis=0) if len(dist_rows) else np.zeros(3, dtype=np.float32)
    dist_std = dist_rows.std(axis=0) if len(dist_rows) else np.zeros(3, dtype=np.float32)
    metrics['Average'] = {
        'CG Dice': float(mean[0]),
        'PZ Dice': float(mean[1]),
        'Prostate Dice': float(mean[2]),
        'CG Dist': float(dist_mean[0]),
        'PZ Dist': float(dist_mean[1]),
        'Prostate Dist': float(dist_mean[2]),
        'Std': {'CG': float(std[0]), 'PZ': float(std[1]), 'Prostate': float(std[2])},
        'Dist Std': {'CG': float(dist_std[0]), 'PZ': float(dist_std[1]), 'Prostate': float(dist_std[2])},
    }
    with open(join(save_dir, '{}_sammed3d_atlas_metrics.json'.format(args.path)), 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=4)
    print('SAM-Med3D atlas CG/PZ Dice: CG={:.4f}+/-{:.4f}, PZ={:.4f}+/-{:.4f}, WG={:.4f}+/-{:.4f}'.format(\
        mean[0], std[0], mean[1], std[1], mean[2], std[2]))
    return mean, std


def medsam_prompt_test(args):
    """Evaluate pretrained SAM/MedSAM with prompts derived from each image mask.

    This intentionally does not use an atlas. It is an oracle-prompt baseline for checking
    whether promptable pretrained segmentation is worth replacing the prompt source with a
    registered atlas later.
    """
    from segment_anything import SamPredictor, sam_model_registry

    device = setup_device(args.gpus)
    if not args.sam_checkpoint:
        raise ValueError('--sam_checkpoint is required for model_type=medsam_prompt')

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    data_path = join(args.data_root, args.path)
    loader = get_dataloader(data_path, 'test', 1, args.crop_size, istest=True, spacing=args.spacing)
    save_dir = join(args.save_dir, args.load_from_dir or '{}_{}'.format(args.model_type, args.path))
    os.makedirs(save_dir, exist_ok=True)
    visual_dir = join(save_dir, '{}_medsam_prompt'.format(args.path))
    os.makedirs(visual_dir, exist_ok=True)
    print(visual_dir)
    metrics = {}
    dices = []
    for i, data in enumerate(loader):
        image = _to_numpy(data['img'])[0, 0].astype(np.float32)
        label = _to_numpy(data['seg'])[0]
        prompt_mask = (label > 0).astype(np.uint8)

        pred = predict_volume_with_sam(predictor, image, prompt_mask, args)
        dice = float(dice_binary(pred, prompt_mask))
        dices.append(dice)
        key = data['key'][0] if isinstance(data['key'], (list, tuple)) else str(data['key'])
        metrics[key] = {'Prostate Dice': dice}
        print('{}: {}, prostate Dice={:.4f}'.format(i, key, dice))

        if args.save_output:
            affine = _to_numpy(data['affine'])[0] if torch.is_tensor(data['affine']) else data['affine'][0]
            nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), join(visual_dir, '{}_pred.nii.gz'.format(key)))
            z = int(np.median(np.where(prompt_mask > 0)[-1])) if prompt_mask.sum() > 0 else image.shape[-1] // 2
            concat = np.concatenate([
                np.rot90(sam_slice_image(image[..., z])[..., 0], k=3),
                127 * np.rot90(pred[..., z].astype(np.uint8), k=3),
                127 * np.rot90(prompt_mask[..., z].astype(np.uint8), k=3),
            ], axis=1)
            cv2.imwrite(join(visual_dir, '{}_z{}.png'.format(key, z)), concat)

    mean = float(np.mean(dices)) if dices else 0.0
    std = float(np.std(dices)) if dices else 0.0
    metrics['Average'] = {'Prostate Dice': mean, 'Std': std}
    with open(join(save_dir, '{}_medsam_prompt_metrics.json'.format(args.path)), 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=4)
    print('MedSAM prompt prostate Dice: {:.4f} +/- {:.4f}'.format(mean, std))
    return mean, std


def setup_device(gpu_id: int) -> str:
    """Setup computing device (GPU/CPU)."""
    return 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu'


def get_dataloader(path: str, phase: str, batch_size: int, crop_size=None, istest=False, spacing=None) -> DataLoader:
    """Return a DataLoader for training or testing."""
    dataset = dataset_loaders(path, 
                 phase, batch_size=batch_size, 
                 np_var='vol', add_batch_axis=True, pad_shape=None,
                resize_factor=spacing, crop_size=crop_size, istest=istest, 
                transform=None, ifbin=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def setup_model(args) -> torch.nn.Module:
    """Initialize the model based on user-defined parameters."""
    from monai.networks import nets
    if args.model_type == 'unet3d':
        model = nets.UNet(spatial_dims=3, in_channels=args.inch, out_channels=args.outch,
                    channels=( 16, 32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2), num_res_units=2, dropout=0.15)
    elif args.model_type == 'deeplabv3':
        from deeplab import Deeplab
        model = Deeplab(3, in_channels=args.inch, out_channels=args.outch)
    elif args.model_type == 'swinunet':
        model = nets.SwinUNETR(spatial_dims=3, in_channels=args.inch, out_channels=args.outch, 
                               img_size=(192, 192, 96), feature_size=48, use_checkpoint=True,)
    elif args.model_type == 'prompt_unet':
        from prompt_unet import PromptedUNet3D
        model = PromptedUNet3D(in_channels=args.inch, out_channels=args.outch)
    return model.to(setup_device(args.gpus))


def save_config(args, save_dir: str):
    """Save training configuration to a JSON file."""
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

def setup_logger(save_dir: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(join(save_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (logs to the console in real-time)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def train(args):
    """Training function."""
    device = setup_device(args.gpus)
    save_dir = join(args.save_dir, '{}_{}'.format(args.model_type, args.path))
    os.makedirs(save_dir, exist_ok=True)
    
    if SummaryWriter is None:
        raise ImportError('tensorboard is required for the standard training path; install tensorboard or use model_type=sammed3d_atlas')
    writer = SummaryWriter(log_dir=join(save_dir, args.log_dir))  # TensorBoard logger
    data_path = join(args.data_root, args.path)
    print('data path is ', data_path)
    train_dataloader = get_dataloader(data_path, 'train', args.batch_size, args.crop_size, args.spacing)

    save_config(args, save_dir)
    logger = setup_logger(save_dir)

    # Model setup
    model = setup_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    from loss_fun import loss_fn
    criterion = loss_fn(args.loss_type)
    
    # Load pretrained weights if resuming training
    if args.continue_train:
        model_path = join(save_dir, 'epoch_{}.pth'.format(args.epoch_load))
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        best_acc = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['img'].to(device), data['seg'].to(device)
            # print(inputs.shape, labels.shape)
            
            optimizer.zero_grad()
            outputs = model(inputs) if args.model_type != 'prompt_unet' else model(inputs, prompt)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % args.print_freq == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
                logger.info('Epoch {}, Batch {}, Loss: {:.4f}'.format(epoch, i, loss.item()))

                zlice = np.median(np.where(labels.detach().cpu() > 0)[-1]).astype(int) if labels.sum() > 0 else 48
                input_show = inputs[..., zlice].cpu()
                for c in range(args.inch):
                    writer.add_images('Input{}'.format(c), input_show[:,c,:,:][:,None,:,:].repeat(1,3,1,1), epoch * len(train_dataloader) + i,
                                  dataformats='NCHW')
                
                writer.add_images('Output', outputs.argmax(dim=1, keepdim=True).cpu().repeat(1, 3, 1, 1, 1)[..., zlice],
                                  epoch * len(train_dataloader) + i, dataformats='NCHW')
                writer.add_images('Ground Truth', labels.cpu()[:, None, ...].repeat(1, 3, 1, 1, 1)[..., zlice],
                                  epoch * len(train_dataloader) + i, dataformats='NCHW')

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info('Epoch {}, Average Loss: {:.4f}'.format(epoch, avg_epoch_loss))
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

        if (epoch + 1) % args.save_freq == 0:
            dice, std = validate(args, model)
            # if np.mean(dice[1:]) > best_acc:
            if dice[1] > best_acc:
                best_acc = np.mean(dice[1:])
                torch.save(model.state_dict(), join(save_dir, 'best.pth'))
                logger.info('Best model saved at epoch {}'.format(epoch))
            logger.info('Dice Score: {}, Std: {}'.format(dice, std))
    torch.save(model.state_dict(), join(save_dir, 'epoch_{}.pth'.format(epoch)))

    writer.close()

# Convert NumPy arrays to Python lists before saving
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy float to Python float
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively process dict
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]  # Recursively process list
    else:
        return obj  # Return as is if already serializable



def validate(args, model, save_output: bool = False, istest=False):
    """Validation function to compute Dice score."""
    model.eval()
    device = setup_device(args.gpus)
    data_path = join(args.data_root, args.path)
    print(data_path)
    test_metrics = {}
    metrics_all = []
    
    from loss_fun import evaluation_metrics as eval_metrics
    em = eval_metrics(classes=args.outch, spacing= args.spacing)
    if istest:
        phase = 'test'
        dice_path = join(args.save_dir, args.load_from_dir, '{}_eval_metrics.json'.format(args.path))
    else:
        phase = 'val'
        dice_path = join(args.save_dir, '{}_{}'.format(args.model_type, args.path), 'eval_metrics.json')
    test_dataloader = get_dataloader(data_path, 'test', args.batch_size, args.crop_size, istest=True)

    with open(dice_path, 'w') as f:
        for i, data in enumerate(test_dataloader):
            inputs, labels = data['img'].to(device), data['seg'].to(device)
            # output is logits, do softmax before putting into dice function
            if i >100:
                break
            with torch.no_grad():
                outputs = model(inputs) if args.model_type != 'prompt_unet' else model(inputs, prompt)
                # print(f'outputs shape: {outputs.shape}')
                outputs = torch.softmax(outputs, dim=1)
                dice = em.dice_coefficient(outputs, labels).cpu().numpy()

                outputs_show = outputs.argmax(dim=1)
                outputs_01 = torch.zeros_like(outputs_show)    
                outputs_01[outputs_show > 0] = 1
                outputs_bin = torch.nn.functional.one_hot(outputs_01, num_classes=2).permute(0,4,1,2,3)
                
                pro_dice= em.dice_coefficient(outputs_bin, (labels>0)).cpu().numpy() 
                print("dice: {}".format(pro_dice))
                dist = em.distance_map(outputs, labels).cpu().numpy()

                test_metrics['Image_{}'.format(i)] = {'Dice': dice[1:], 'Prostate Dice': pro_dice[1:], 'Distance': dist}
                metrics_all.append(list(dice[1:]) + list(pro_dice[1:]) + list(dist[1:]))
                
                # print(f'Dice Score: {dice}')
                if save_output:
                    
                    concat_image = np.concatenate([255*np.rot90(inputs.cpu().numpy()[0,0,:,:,48], k=3),
                                                   127*np.rot90(outputs_show.cpu().numpy()[0,:,:,48], k=3),
                                                     127*np.rot90(labels.cpu().numpy()[0,:,:,48], k=3)], axis=1)
                    
                    __save = join(args.save_dir, '{}/{}_visual'.format(args.load_from_dir, args.path))
                    os.makedirs(__save, exist_ok=True)
                    cv2.imwrite(join(__save, '{}.png'.format(data['key'][0])), concat_image)
                    save_img = nib.Nifti1Image(outputs_show.cpu().numpy().astype(float)[0], affine=data['affine'][0],)
                    
                    nib.save(save_img, join(__save, data['key'][0]+'.nii.gz'))
 
        metrics_all = np.vstack(np.array(metrics_all))
        mean_dice, mean_std = np.mean(metrics_all, axis=0), np.std(metrics_all, axis=0)
        # print(f'Mean Dice: {mean_dice}, Std: {mean_std}')
        test_metrics['Average'] = {'Dice': list(mean_dice), 'Std': list(mean_std)}

        test_metrics = convert_to_serializable(test_metrics)
        json.dump(test_metrics, f, indent=4)
     
    return mean_dice, mean_std

def test(args):
    """Testing function."""
    device = setup_device(args.gpus)
    save_dir = join(args.save_dir, args.load_from_dir)
    model_path = join(save_dir, 'best.pth')

    model = setup_model(args)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    args.batch_size = 1
    dice, std = validate(args, model, save_output=True, istest=True)
    print('Dice Score: {}, Std: {}'.format(dice, std))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="3D UNet Prostate Segmentation")
    
    # General arguments
    parser.add_argument('--data_root', type=str, default='../../Datasets/ProstateDatasets/data_split_files', help='Path to data')
    parser.add_argument('--path', type=str, default='4-prostate158', help='Path to data')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train', help='Train or test phase')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--spacing', type=float, nargs='+', default=[0.5, 0.5, 1.0], help='Spacing for resizing images')
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'dice', 'focal'], default='dice',
                        help='Loss function')
    parser.add_argument('--inch', type=int, default=1, help='Number of input channels')
    parser.add_argument('--outch', type=int, default=3, help='Number of output channels')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--epoch_load', default=100, help='Epoch number for loading model')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every X epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--crop_size', type=int, default=None, help='Crop size')
    parser.add_argument('--gpus', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--continue_train', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--model_type', type=str, default='unet3d', help='Model type')
    parser.add_argument('--sam_checkpoint', type=str, default='', help='Path or URL to pretrained SAM/MedSAM/SAM-Med3D checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['default', 'vit_h', 'vit_l', 'vit_b'], help='SAM backbone type')
    parser.add_argument('--atlas_path', type=str, default='../register/atlas/atlas9.npz', help='Fallback atlas npz containing atlas and mask_atlas labels with PZ=1 and CG=2')
    parser.add_argument('--pretrained_light_register', type=str,
                        default='/cluster/project7/longitude/atlasSam/register/checkpoints/light_register/latest_light_register.pth',
                        help='Frozen lightweight atlas-to-image registration checkpoint used to align atlas masks before MedSAM training/test; empty disables it')
    parser.add_argument('--atlas_bank_dir', type=str, default='../register/atlas_bank', help='Directory containing reference_bank.csv with selected atlas masks')
    parser.add_argument('--atlas_bank_data_root', type=str, default=DEFAULT_ATLAS_BANK_DATA_ROOT, help='Readable dataset root used to remap stale atlas-bank CSV paths')
    parser.add_argument('--atlas_bank_mode', type=str, default='cycle', choices=['first', 'cycle', 'random', 'ensemble', 'topk'], help='How to use selected atlas masks from atlas_bank_dir')
    parser.add_argument('--atlas_top_k', type=int, default=5, help='Number of similar atlases to retrieve for atlas_bank_mode=topk')
    parser.add_argument('--atlas_retrieval_features', type=str, default='whole_t2_mean,whole_t2_std,whole_t2_median', help='Comma-separated atlas-bank numeric features used for top-k retrieval')
    parser.add_argument('--atlas_similarity_temperature', type=float, default=1.0, help='Temperature for converting retrieval distance to similarity weight')
    parser.add_argument('--atlas_fusion_threshold', type=float, default=0.5, help='Weighted score threshold for final top-k atlas fusion')
    parser.add_argument('--pz_label', type=int, default=1, help='PZ label value in atlas-bank prostate_zones masks')
    parser.add_argument('--cg_label', type=int, default=2, help='CG label value in atlas-bank prostate_zones masks')
    parser.add_argument('--sam_crop_size', type=int, default=128, help='Cubic ROI size for SAM-Med3D inference')
    parser.add_argument('--atlas_prompt_pad', type=int, default=8, help='Reserved padding around atlas prompt ROI')
    parser.add_argument('--sam_prob_threshold', type=float, default=0.5, help='Probability threshold for SAM-Med3D binary masks')
    parser.add_argument('--sam_finetune_checkpoint', type=str, default='', help='Fine-tuned SAM-Med3D checkpoint to load for resume/test')
    parser.add_argument('--medsam_bottleneck_channels', type=int, default=32, help='Hidden channels for the trainable bottleneck adapter on frozen SAM-Med3D image embeddings')
    parser.add_argument('--bce_loss_weight', type=float, default=1.0, help='BCE loss weight for SAM-Med3D fine-tuning')
    parser.add_argument('--dice_loss_weight', type=float, default=1.0, help='Dice loss weight for SAM-Med3D fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay for SAM-Med3D fine-tuning')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm for SAM-Med3D fine-tuning')
    parser.add_argument('--prompt_type', type=str, default='box', choices=['box', 'points', 'box_points'], help='Prompt generated from the image mask')
    parser.add_argument('--num_prompt_points', type=int, default=1, help='Number of positive point prompts per non-empty slice')
    parser.add_argument('--box_pad', type=int, default=3, help='Padding in pixels around mask-derived box prompt')
    parser.add_argument('--min_prompt_pixels', type=int, default=10, help='Skip slices with fewer foreground pixels')
    parser.add_argument('--multimask_output', action='store_true', help='Ask SAM for multiple masks and keep the highest-score mask')
    parser.add_argument('--save_output', action='store_true', help='Save predicted masks and visual slices during test')
    # Testing
    parser.add_argument('--load_from_dir', type=str, default='unet3d_picai_zones_ratio0.7', help='Directory to load model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    '''
    usage: test: python segment.py --path picai_zones_ratio0.7 --epoch_load best --phase test --load_from_dir unet3d_1-uclH-data_ratio0.7
           train: python segment.py --path 1-uclH-data_ratio0.8 --phase train
    '''
    if args.model_type == 'medsam_prompt':
        if args.phase != 'test':
            raise ValueError('model_type=medsam_prompt is an inference-only baseline; use --phase test')
        medsam_prompt_test(args)
    elif args.model_type == 'sammed3d_atlas':
        if args.phase == 'train':
            atlas_conditioned_sammed3d_train(args)
        elif args.phase == 'test':
            atlas_conditioned_sammed3d_test(args)
    elif args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
