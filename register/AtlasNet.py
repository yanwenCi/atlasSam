# reglite3d.py
# Lightweight 3D atlas->image registration for T2 prostate (affine + SVF)
# Outputs a sampling grid to warp atlas intensity and CG/PZ masks.
# Author: you | License: MIT

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Small building blocks
# ---------------------------
class DoubleConv3D(nn.Module):
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

class UNetSVF3D(nn.Module):
    """ Shallow UNet predicting SVF (3 channels). """
    def __init__(self, in_ch=2, base=16):
        super().__init__()
        self.enc1 = DoubleConv3D(in_ch, base)          # 192x192x96
        self.pool1 = nn.MaxPool3d(2)                   # 96x96x48
        self.enc2 = DoubleConv3D(base, base*2)
        self.pool2 = nn.MaxPool3d(2)                   # 48x48x24
        self.enc3 = DoubleConv3D(base*2, base*4)

        self.up2  = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv3D(base*4, base*2)
        self.up1  = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv3D(base*2, base)

        self.svf  = nn.Conv3d(base, 3, 3, padding=1)   # SVF (vx,vy,vz)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        v  = self.svf(d1)
        return v

class AffineHead3D(nn.Module):
    """ Tiny affine regressor from joint (atlas,image) volume. """
    def __init__(self, in_ch=2, ch=16):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv3d(in_ch, ch, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch*2, 3, stride=2, padding=1),  nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1),nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch*4, 64), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 12)  # 9 for A, 3 for b
        )
    def forward(self, atlas_t2, image_t2):
        x = torch.cat([atlas_t2, image_t2], dim=1)
        p = self.fc(self.feat(x))
        A = p[:, :9].view(-1,3,3)
        b = p[:, 9:].view(-1,3,1)
        # initialise near identity
        A = A + torch.eye(3, device=A.device).view(1,3,3)
        return A, b

# ---------------------------
# Grids & warps
# ---------------------------
def idgrid(shape, device, dtype=torch.float32):
    B, _, D, H, W = shape
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
    g = torch.stack([x, y, z], dim=-1).unsqueeze(0).repeat(B,1,1,1,1)  # (B,D,H,W,3)
    return g

def warp(img, grid, mode='bilinear'):
    return F.grid_sample(img, grid, mode=mode, padding_mode='border', align_corners=True)

def compose(grid_a, grid_b):
    """ Compose two normalized grids: result = grid_b ∘ grid_a. """
    return F.grid_sample(grid_b.permute(0,4,1,2,3), grid_a, align_corners=True, padding_mode='border').permute(0,2,3,4,1)

def affine_grid(A, b, vol_shape):
    B, _, D, H, W = vol_shape
    g = idgrid(vol_shape, A.device, A.dtype)                 # [-1,1]
    gv = g.view(B, -1, 3).transpose(1,2)                     # (B,3,N)
    gv = (A @ gv + b).transpose(1,2).view(B, D, H, W, 3)     # linear approx in norm coords
    return gv.clamp(-1, 1)

def s_and_s(v, steps=7):
    """ Exponential map of SVF via scaling-and-squaring in normalized coords. """
    B, C, D, H, W = v.shape
    disp = v.permute(0,2,3,4,1) / (2**steps)   # (B,D,H,W,3)
    phi = idgrid((B,1,D,H,W), v.device, v.dtype) + disp
    for _ in range(steps):
        phi = compose(phi, phi)
    return phi

def jac_det(grid):
    dx = grid[:,:,:,1:,:] - grid[:,:,:,:-1,:]
    dy = grid[:,:,1:,:,:] - grid[:,:,:-1,:,:]
    dz = grid[:,1:,:,:,:] - grid[:,:-1,:,:,:]
    print( grid.shape, dx.shape, dy.shape, dz.shape )
    def padx(t): return F.pad(t, (0,0,0,0,0,0,0,0,0,1))
    def pady(t): return F.pad(t, (0,0,0,0,0,0,0,1,0,0))
    def padz(t): return F.pad(t, (0,0,0,0,0,0,1,0,0,0))
    dx, dy, dz = padx(dx), pady(dy), padz(dz)
    print( grid.shape, dx.shape, dy.shape, dz.shape )
    J = torch.stack([dx, dy, dz], dim=-1)  # (B,D,H,W,3,3)
    det = (
        J[...,0,0]*(J[...,1,1]*J[...,2,2]-J[...,1,2]*J[...,2,1])
        - J[...,0,1]*(J[...,1,0]*J[...,2,2]-J[...,1,2]*J[...,2,0])
        + J[...,0,2]*(J[...,1,0]*J[...,2,1]-J[...,1,1]*J[...,2,0])
    )
    return det

# ---------------------------
# Losses
# ---------------------------
class LNCC3D(nn.Module):
    def __init__(self, win=(9,9,9), eps=1e-5):
        super().__init__()
        self.win = win; self.eps = eps
    def forward(self, I, J):
        B, C, D, H, W = I.shape
        pad = tuple(w//2 for w in self.win)
        filt = torch.ones((1,1,*self.win), device=I.device, dtype=I.dtype)
        def conv(x): return F.conv3d(x, filt, padding=pad)
        WN = float(self.win[0]*self.win[1]*self.win[2])
        muI, muJ = conv(I)/WN, conv(J)/WN
        I2, J2, IJ = conv(I*I)/WN, conv(J*J)/WN, conv(I*J)/WN
        sI2, sJ2 = I2 - muI*muI, J2 - muJ*muJ
        sIJ = IJ - muI*muJ
        ncc = sIJ*sIJ / (sI2*sJ2 + self.eps)
        return ncc.mean()

@dataclass
class RegWeights:
    lambda_smooth: float = 1e-4
    lambda_jac: float    = 1e-3
    w_tz: float          = 2.0   # weight TZ more than CG in Dice
    w_cg: float          = 1.0

def dice_soft(pred, tgt, eps=1e-6):
    inter = (pred * tgt).sum(dim=(2,3,4))
    den   = pred.sum(dim=(2,3,4)) + tgt.sum(dim=(2,3,4))
    return (2*inter + eps) / (den + eps)  # (B, C)


def inv_grid_total(A: torch.Tensor,
                   b: torch.Tensor,
                   v: torch.Tensor,
                   vol_shape: torch.Size,
                   steps: int = 7) -> torch.Tensor:
    """
    Build the image→atlas grid (the inverse of the forward atlas→image warp).
    
    Inputs:
      A:          (B,3,3)  affine matrix (atlas→image)
      b:          (B,3,1)  affine translation (atlas→image)
      v:          (B,3,D,H,W) stationary velocity field (SVF) for exp(v) (atlas→image)
      vol_shape:  torch.Size like (B,1,D,H,W) of the *image* volume
      steps:      scaling-and-squaring steps (same as used in forward)

    Returns:
      grid_inv:   (B,D,H,W,3) normalized grid mapping image coords → atlas coords
                  usable with F.grid_sample(image, grid_inv, align_corners=True)
    
    Math:
      Forward total warp (atlas→image): Φ = Φ_aff ∘ exp(v)
      Its inverse (image→atlas):        Φ^{-1} = exp(-v) ∘ Φ_aff^{-1}
      With compose(a,b) = b ∘ a, we need:
         grid_inv = compose( grid_svf_inv , grid_aff_inv )
    """
    device = A.device
    dtype  = A.dtype

    # affine inverse (image→atlas)
    A_inv, b_inv = invert_affine(A, b)
    grid_aff_inv = affine_grid(A_inv, b_inv, vol_shape)  # (B,D,H,W,3)

    # diffeo inverse via SVF: exp(-v)  (atlas→image used exp(+v))
    grid_svf_inv = s_and_s(-v, steps=steps)              # (B,D,H,W,3)

    # IMPORTANT: compose order!
    # compose(a,b) = b ∘ a  ⇒  compose(grid_svf_inv, grid_aff_inv) = grid_aff_inv ∘ grid_svf_inv
    # which equals exp(-v) ∘ Φ_aff^{-1} as desired (image→atlas).
    grid_inv = compose(grid_svf_inv, grid_aff_inv)

    return grid_inv


def dice_multiclass_hard(pred_oh: torch.Tensor, tgt_oh: torch.Tensor, C: int, ignore_bg: bool = True, eps: float = 1e-6):
    """
    pred_lab, tgt_lab: (B,1,D,H,W) integer labels in {0..C-1}
    Computes per-class and mean Dice (optionally excluding background class 0).
    """
    B = pred_oh.size(0)

    # pred_oh = F.one_hot(pred_lab.squeeze(1).long(), num_classes=C).permute(0,4,1,2,3).float()  # (B,C,D,H,W)
    # tgt_oh  = F.one_hot(tgt_lab.squeeze(1).long(),  num_classes=C).permute(0,4,1,2,3).float()

    inter = (pred_oh * tgt_oh).sum(dim=(0,2,3,4))                       # (C,)
    den   = pred_oh.sum(dim=(0,2,3,4)) + tgt_oh.sum(dim=(0,2,3,4))      # (C,)
    dice_c = (2*inter + eps) / (den + eps)                               # (C,)

    if ignore_bg and C > 1:
        dice_mean = dice_c[1:].mean()
    else:
        dice_mean = dice_c.mean()
    return dice_c, dice_mean



def laplacian3d(v: torch.Tensor, replicate_pad: bool = True) -> torch.Tensor:
    """
    v: (B, C, D, H, W) — e.g., your SVF with C=3
    returns Laplacian(v) with same shape
    """
    B, C, D, H, W = v.shape
    # depthwise kernel: center -6, 6-neighbors = +1
    k = torch.zeros((C, 1, 3, 3, 3), device=v.device, dtype=v.dtype)
    k[:, :, 1, 1, 1] = -6.0
    k[:, :, 1, 1, 0] = 1.0  # x-
    k[:, :, 1, 1, 2] = 1.0  # x+
    k[:, :, 1, 0, 1] = 1.0  # y-
    k[:, :, 1, 2, 1] = 1.0  # y+
    k[:, :, 0, 1, 1] = 1.0  # z-
    k[:, :, 2, 1, 1] = 1.0  # z+

    if replicate_pad:
        v_pad = F.pad(v, (1, 1, 1, 1, 1, 1), mode='replicate')  # (W_l,W_r,H_l,H_r,D_l,D_r)
        out = F.conv3d(v_pad, k, padding=0, groups=C)
    else:
        out = F.conv3d(v, k, padding=1, groups=C)  # zero-pad
    return out

# ---------------------------
# The small registration net
# ---------------------------
class RegLite3D(nn.Module):
    """
    atlas_t2, image_t2: (B,1,D,H,W) in same space (e.g., 192x192x96).
    atlas_probs: dict {'CG':(B,1,D,H,W), 'PZ':(B,1,D,H,W)} in atlas space
    Returns:
      grids (aff, svf, total) and warped atlas/probs in image space.
    """
    def __init__(self, steps=7, base=16, weights: RegWeights=RegWeights()):
        super().__init__()
        self.aff = AffineHead3D(in_ch=2)
        self.svf = UNetSVF3D(in_ch=2, base=base)
        self.lncc = LNCC3D(win=(9,9,9))
        self.steps = steps
        self.w = weights

    def forward(self, atlas_t2, image_t2, atlas_probs: Optional[Dict[str, torch.Tensor]]=None):
        # 1) affine
        A, b = self.aff(atlas_t2, image_t2)
        g_aff = affine_grid(A, b, atlas_t2.shape)              # (B,D,H,W,3)
        a_aff = warp(atlas_t2, g_aff)                          # warped atlas T2

        # 2) SVF (on [atlas_aff, image])
        v = self.svf(torch.cat([a_aff, image_t2], dim=1))      # (B,3,D,H,W)
        g_svf = s_and_s(v, steps=self.steps)
        g_tot = compose(g_aff, g_svf)

        # Warps
        atlas_warp = warp(atlas_t2, g_tot)
        warped_probs = None
        if atlas_probs is not None:
            warped_probs = {k: warp(vt, g_tot) for k, vt in atlas_probs.items()}

        # Loss terms (return for training)
        lncc_sim = 1.0 - self.lncc(image_t2, atlas_warp)
        # region-weighted Dice to emphasise TZ/CG alignment if probs provided
        dice_reg = 0.0
        if warped_probs is not None:
            # Treat warped priors as "pred", and build "pseudo-target" by thresholding image-driven prior from atlas?
            # Here we encourage sharp overlap with its own warp via sharpening (acts as regulariser):
            pred = torch.cat([warped_probs['PZ'], warped_probs['CG']], dim=1)  # (B,2,D,H,W)
            # sharpen to avoid trivial smooth masks
            pred_s = torch.clamp(pred, 0, 1)
            dice = dice_soft(pred_s, pred_s.detach())  # identity, but lets us weight classes
            dice_w = (self.w.w_tz * dice[:,0] + self.w.w_cg * dice[:,1]) / (self.w.w_tz + self.w.w_cg)
            dice_reg = 1.0 - dice_w.mean()

        # smoothness (bending proxy): Laplacian of v
        lap = laplacian3d(v)
        l_smooth = (lap.pow(2)).mean()

        # jacobian penalty
        detJ = jac_det(g_tot)
        l_jac = F.relu(-torch.log(detJ.clamp_min(1e-6))).mean()

        loss = lncc_sim + self.w.lambda_smooth*l_smooth + self.w.lambda_jac*l_jac + 0.1*dice_reg

        pos_frac = (detJ > 0).float().mean()

        return {
            'loss': loss,
            'terms': {'sim': lncc_sim.detach(), 'smooth': l_smooth.detach(), 'jac': l_jac.detach(), 'pos_jac': pos_frac.detach()},
            'A': A, 'b': b, 'v': v,
            'grid_aff': g_aff, 'grid_svf': g_svf, 'grid_total': g_tot,
            'atlas_warp': atlas_warp,
            'warped_probs': warped_probs,
        }
