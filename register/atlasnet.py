# atlas_reg_model.py
# PyTorch model for atlas→image registration (affine + diffeo SVF)
# Author: (you)
# License: MIT

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Small 3D UNet (SVF head)
# --------------------------
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.InstanceNorm3d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.InstanceNorm3d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet3D_SVF(nn.Module):
    def __init__(self, in_ch=2, base=16, out_ch=3):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.down2 = DoubleConv(base, base*2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = DoubleConv(base*4+base*2, base*2)
        self.up1 = DoubleConv(base*2+base, base)
        self.flow = nn.Conv3d(base, out_ch, 3, padding=1)

    def forward(self, x):
        x1 = self.down1(x)           # D
        x2 = self.down2(self.pool(x1))  # D/2
        x3 = self.down3(self.pool(x2))  # D/4
        u2 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=True)
        u2 = self.up2(torch.cat([u2, x2], dim=1))
        u1 = F.interpolate(u2, scale_factor=2, mode='trilinear', align_corners=True)
        u1 = self.up1(torch.cat([u1, x1], dim=1))
        v = self.flow(u1)  # SVF
        return v


# --------------------------
# Affine predictor (very light)
# --------------------------
class AffineCNN(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        ch = 16
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, ch, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch*2, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch*4, ch*4, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch*4, 64), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 12)  # 9 for A, 3 for b
        )

    def forward(self, atlas_t2, image_t2):
        x = torch.cat([atlas_t2, image_t2], dim=1)
        h = self.features(x)
        p = self.head(h)
        A = p[:, :9].view(-1, 3, 3)
        b = p[:, 9:].view(-1, 3, 1)
        # initialize close to identity
        A = A + torch.eye(3, device=A.device).view(1,3,3)
        return A, b


# --------------------------
# Grids and warping
# --------------------------
def identity_grid(shape, device, dtype=torch.float32):
    B, C, D, H, W = shape
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([x, y, z], dim=-1).unsqueeze(0).repeat(B,1,1,1,1)
    return grid  # (B,D,H,W,3) in [-1,1]

def affine_grid_from_Ab(A, b, shape):
    """Return grid that applies atlas->image affine (normalized coords)."""
    B, C, D, H, W = shape
    grid = identity_grid(shape, A.device, A.dtype)  # (B,D,H,W,3)
    # convert normalized grid to homogeneous voxel coords, apply A,b approximately
    # For simplicity, treat normalized coords as linear space; works if volumes pre-resampled.
    g = grid.view(B, -1, 3).transpose(1,2)  # (B,3,N)
    g = (A @ g + b).transpose(1,2)          # (B,N,3)
    g = g.view(B, D, H, W, 3)
    return g.clamp(-1, 1)

def warp(img, grid, mode='bilinear'):
    return F.grid_sample(img, grid, mode=mode, padding_mode='border', align_corners=True)

def spatial_laplacian(v):
    # simple discrete Laplacian for bending energy proxy
    lap = (
        -6*v
        + F.pad(v, (0,0,0,0,1,0))[:,:,:-1] + F.pad(v, (0,0,0,0,0,1))[:,:,1:]
        + F.pad(v, (0,0,1,0))[:,:,:, :-1] + F.pad(v, (0,0,0,1))[:,:,:, 1:]
        + F.pad(v, (1,0))[:,:,:,:-1] + F.pad(v, (0,1))[:,:,:,1:]
    )
    return lap

def compose_displacement(phi_a, phi_b):
    """Compose two displacement fields in normalized coords:
       phi = phi_b ∘ phi_a. Both are grids in [-1,1] (B,D,H,W,3)."""
    # sample phi_b at locations given by phi_a
    return F.grid_sample(phi_b.permute(0,4,1,2,3), phi_a, align_corners=True, padding_mode='border').permute(0,2,3,4,1)

def scaling_and_squaring(v, steps=7):
    """Exponential map via scaling-and-squaring in normalized coords."""
    # Convert velocity (B,3,D,H,W) to displacement grid step
    B, C, D, H, W = v.shape
    # normalize velocity magnitude w.r.t. steps
    disp = (v.permute(0,2,3,4,1)) / (2**steps)  # (B,D,H,W,3)
    idg = identity_grid((B,1,D,H,W), v.device, v.dtype)
    phi = idg + disp
    for _ in range(steps):
        # phi = phi ∘ phi
        phi = compose_displacement(phi, phi)
    return phi  # (B,D,H,W,3)

def jacobian_det_from_grid(grid):
    """grid: (B,D,H,W,3) in [-1,1]; return det(J) approx."""
    # finite differences
    dx = grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]
    dy = grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]
    dz = grid[:, 1:, :, :, :] - grid[:, :-1, :, :, :]
    # pad to same size
    def pad4(t): return F.pad(t, (0,0,0,0,0,1,0,0,0,0))  # pad along respective axis
    dx, dy, dz = pad4(dx), F.pad(dy, (0,0,0,0,0,0,0,1,0,0)), F.pad(dz, (0,0,0,0,0,0,0,0,0,1))
    # Jacobian 3x3
    J = torch.stack([dx, dy, dz], dim=-1)  # (B,D,H,W,3,3)
    # determinant
    det = (
        J[...,0,0]*(J[...,1,1]*J[...,2,2]-J[...,1,2]*J[...,2,1])
        - J[...,0,1]*(J[...,1,0]*J[...,2,2]-J[...,1,2]*J[...,2,0])
        + J[...,0,2]*(J[...,1,0]*J[...,2,1]-J[...,1,1]*J[...,2,0])
    )
    return det


# --------------------------
# Losses: LNCC, bending, Jacobian penalty
# --------------------------
class LNCC(nn.Module):
    def __init__(self, win: Tuple[int,int,int]=(9,9,9), eps: float=1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        # I,J: (B,1,D,H,W) normalized volumes
        B, C, D, H, W = I.shape
        pad = tuple(w//2 for w in self.win)
        filt = torch.ones((1,1,*self.win), device=I.device, dtype=I.dtype)

        def conv(x):
            return F.conv3d(x, filt, padding=pad)

        I2 = I*I; J2 = J*J; IJ = I*J
        sum_filt = torch.prod(torch.tensor(self.win, device=I.device, dtype=I.dtype))
        mu_I = conv(I)/sum_filt
        mu_J = conv(J)/sum_filt
        sigma_I2 = conv(I2)/sum_filt - mu_I**2
        sigma_J2 = conv(J2)/sum_filt - mu_J**2
        sigma_IJ = conv(IJ)/sum_filt - mu_I*mu_J

        ncc = sigma_IJ**2 / (sigma_I2 * sigma_J2 + self.eps)
        return ncc.mean()  # higher is better


@dataclass
class RegLossWeights:
    lambda_aff: float = 1e-3
    lambda_svf: float = 1e-4
    lambda_jac: float = 1e-3


# --------------------------
# Full model
# --------------------------
class Atlas2ImageRegModel(nn.Module):
    """
    Inputs:
      atlas_t2: (B,1,D,H,W)
      image_t2: (B,1,D,H,W)
    Optionally:
      atlas_probs: dict {'CG': (B,1,D,H,W), 'PZ': (B,1,D,H,W)}
    Returns:
      dict with warped atlas, grids, losses, and warped priors.
    """
    def __init__(self, lncc_win=(9,9,9), steps=7, loss_w: RegLossWeights=RegLossWeights()):
        super().__init__()
        self.affine = AffineCNN(in_ch=2)
        self.svf = UNet3D_SVF(in_ch=2, base=16, out_ch=3)
        self.lncc = LNCC(win=lncc_win)
        self.steps = steps
        self.w = loss_w

    def forward(self, atlas_t2, image_t2, atlas_probs: Optional[Dict[str, torch.Tensor]]=None):
        # 1) Affine
        A, b = self.affine(atlas_t2, image_t2)  # (B,3,3), (B,3,1)
        grid_aff = affine_grid_from_Ab(A, b, atlas_t2.shape)  # (B,D,H,W,3)
        atlas_aff = warp(atlas_t2, grid_aff)

        # 2) SVF (diffeo) on concatenated (atlas_aff, image)
        v = self.svf(torch.cat([atlas_aff, image_t2], dim=1))  # (B,3,D,H,W)
        grid_svf = scaling_and_squaring(v, steps=self.steps)   # (B,D,H,W,3)
        # compose affine then svf: grid_total = grid_svf ∘ grid_aff
        grid_total = compose_displacement(grid_aff, grid_svf)

        atlas_warp = warp(atlas_t2, grid_total)

        # Warp probability priors if provided
        warped_probs = None
        if atlas_probs is not None:
            warped_probs = {k: warp(vt, grid_total) for k, vt in atlas_probs.items()}

        # Losses
        # Similarity on final warp
        lncc_val = self.lncc(image_t2, atlas_warp)
        loss_sim = 1.0 - lncc_val

        # Affine regulariser (stay near identity)
        I3 = torch.eye(3, device=A.device, dtype=A.dtype).view(1,3,3)
        loss_aff = (A - I3).pow(2).mean() + b.pow(2).mean()

        # Bending energy on SVF via Laplacian proxy
        lap = spatial_laplacian(v)
        loss_smooth = (lap.pow(2)).mean()

        # Soft Jacobian penalty (folding)
        detJ = jacobian_det_from_grid(grid_total)
        loss_jac = F.relu(-torch.log(detJ.clamp_min(1e-6))).mean()

        loss = loss_sim + self.w.lambda_aff*loss_aff + self.w.lambda_svf*loss_smooth + self.w.lambda_jac*loss_jac

        # Topology metric
        pos_frac = (detJ > 0).float().mean()

        return {
            'loss': loss,
            'loss_terms': {
                'sim': loss_sim.detach(),
                'aff': loss_aff.detach(),
                'smooth': loss_smooth.detach(),
                'jac': loss_jac.detach(),
                'lncc': lncc_val.detach(),
                'pos_jac_frac': pos_frac.detach(),
            },
            'A': A, 'b': b,
            'v': v,
            'grid_aff': grid_aff,
            'grid_svf': grid_svf,
            'grid_total': grid_total,
            'atlas_warp': atlas_warp,
            'warped_probs': warped_probs
        }

