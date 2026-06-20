
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def loss_fn(loss_type):
    """
    Returns the loss function based on the loss type.

    Args:
        loss_type (str): Name of the loss function.

    Returns:
        loss_fn (nn.Module): Loss function.
    """
    if loss_type == 'ce':
        return CrossEntropyLoss3D()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss3D()
    elif loss_type.lower() == 'ncc':
        return NCC().loss
    elif loss_type.lower() == 'mse':
        # return nn.MSELoss()
        return F.mse_loss
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

class NCC():
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

def global_mutual_information(t1, t2):
    bin_centers = torch.linspace(0.0, 1.0, 21)

    if t1.is_cuda:
        bin_centers = bin_centers.cuda()

    num_bins = bin_centers.shape[0]
    sigma_ratio = 0.5
    eps = 1.19209e-07
    
    sigma = torch.mean(bin_centers[1:]-bin_centers[:-1])*sigma_ratio
    preterm = 1 / (2 * torch.square(sigma))
    
    if len(t1.shape) == 3:
        w, h, z = t1.shape
        c, batch = 1, 1
    elif len(t1.shape) == 4:
        c, w, h, z = t1.shape
        batch = 1 
    elif len(t1.shape) == 5:
        batch, c, w, h, z = t1.shape
    else:
        raise NotImplementedError 

    t1 = torch.reshape(t1, [batch, c*w*h*z, 1])
    t2 = torch.reshape(t2, [batch, c*w*h*z, 1])
    nb_voxels = t1.shape[1] * 1.0
    vbc = torch.reshape(bin_centers, (1, 1, -1))

    I_a = torch.exp(- preterm * torch.square(t1 - vbc))
    I_a = I_a / torch.sum(I_a, -1, keepdim=True)
    I_a_permute = torch.transpose(I_a, 2, 1)
    
    I_b = torch.exp(- preterm * torch.square(t2 - vbc))
    I_b = I_b / torch.sum(I_b, -1, keepdim=True)
    I_b_permute = torch.transpose(I_b, 2, 1)

    pa = torch.mean(I_a, axis=1, keepdim=True)
    pb = torch.mean(I_b, axis=1, keepdim=True)
    pa = torch.transpose(pa, 2, 1)

    papb = torch.matmul(pa, pb) + eps
    pab = torch.matmul(I_a_permute, I_b) / nb_voxels

    return torch.mean(torch.sum(pab*torch.log(pab/papb + eps), dim=[1, 2]))

    
def gradient_dx(arr):
    return (arr[:, 2:, 1:-1, 1:-1] - arr[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(arr):
    return (arr[:, 1:-1, 2:, 1:-1] - arr[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(arr):
    return (arr[:, 1:-1, 1:-1, 2:] - arr[:, 1:-1, 1:-1, :-2]) / 2


def gradient_txyz(Txyz, fn):
    return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], axis=1)


def bending_energy(ddf):
    # 1st order
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)

    # 2nd order
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)

    return torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)



def dice_coefficient(pred, target, smooth=1e-6):
    """
    Computes the Dice coefficient for a single class.

    Args:
        pred (Tensor): Predicted tensor (B, C, D, H, W).
        target (Tensor): Ground truth tensor (B, C, D, H, W), one-hot encoded.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        dice (Tensor): Computed Dice coefficient.
    """
    # Flatten the tensors (B, C, D, H, W) -> (B, C, N)
    # pred = torch.softmax(pred, dim=1)
    target= target.squeeze(1)
    target = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3)
    pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1)

    # Compute Dice coefficient
    intersection = (pred * target).sum(dim=2)
    dice_coeff = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)
    dice_coeff = torch.mean(dice_coeff.mean(dim=0)[1:])
    return dice_coeff


class CrossEntropyLoss3D(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss3D, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Computes the Cross-Entropy Loss for 3D images.

        Args:
            pred (Tensor): Predicted logits (B, C, D, H, W).
            target (Tensor): Ground truth tensor (B, D, H, W), containing class indices.

        Returns:
            loss (Tensor): Computed cross-entropy loss.
        """
        # Ensure target shape is correct (B, D, H, W)
        target = target.squeeze(1)
        assert pred.shape[2:] == target.shape[1:], "Pred and target shapes must match except for the class dimension"

        # Compute cross-entropy loss
        loss = self.ce_loss(pred, target.long())  # Target should contain class indices (not one-hot)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Computes the Dice Loss for 3D images.

        Args:
            pred (Tensor): Predicted tensor (B,  D, H, W) with probabilities.
            target (Tensor): Ground truth tensor (B,  D, H, W), one-hot encoded.
        
        Returns:
            dice_loss (Tensor): Computed dice loss.
        """
        # Flatten the tensors (B, C, D, H, W) -> (B, C, N)

        # Compute Dice coefficient
       
        pred = pred.round().squeeze(1)
        target = target.squeeze(1)
        target = F.one_hot(target.long(), num_classes=3).permute(0, 4, 1, 2, 3)
        pred = F.one_hot(pred.long(), num_classes=3).permute(0, 4, 1, 2, 3)
        intersection = torch.sum(pred * target, dim=[2,3,4])
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum(dim=[2,3,4]) + target.sum(dim=[2,3,4]) + self.smooth)
        # print(dice_coeff.mean())
        # Dice loss (1 - mean Dice coefficient over batch)
        return 1 - dice_coeff.mean()


class FocalLoss3D(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for 3D image segmentation.
        
        Args:
            alpha (Tensor, optional): Class-wise weights (C,).
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss3D, self).__init__()
        self.alpha = alpha  # If None, equal weighting
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute focal loss.

        Args:
            pred (Tensor): Logits (B, C, D, H, W).
            target (Tensor): Ground truth class indices (B, D, H, W).

        Returns:
            loss (Tensor): Computed focal loss.
        """
        # Convert logits to probabilities using softmax
        pred_prob = F.softmax(pred, dim=1)  # (B, C, D, H, W)

        # Gather the probabilities corresponding to the ground truth class
        target_one_hot = F.one_hot(target.long(), pred.shape[1]).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        prob = (pred_prob * target_one_hot).sum(dim=1)  # (B, D, H, W) -> prob for correct class

        # Compute focal loss
        focal_weight = (1 - prob) ** self.gamma  # Apply focusing
        ce_loss = -torch.log(prob.clamp(min=1e-5))  # Cross-entropy loss
        focal_loss = focal_weight * ce_loss  # Apply focal weight

        # Apply class balancing (if alpha is provided)
        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, target.view(-1)).view(target.shape)
            focal_loss *= alpha_factor

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # Returns per voxel loss


class evaluation_metric():
    def __init__(self):
        
        pass

    def dice_coefficient(self, pred, target, smooth=1e-6):
        """
        Computes the Dice coefficient for a single class.

        Args:
            pred (Tensor): Predicted tensor (B, C, D, H, W).
            target (Tensor): Ground truth tensor (B, C, D, H, W), one-hot encoded.
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            dice (Tensor): Computed Dice coefficient.
        """
        # Flatten the tensors (B, C, D, H, W) -> (B, C, N)
        pred = torch.softmax(pred, dim=1)
        target = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3)
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=2)
        dice_coeff = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)
        dice_coeff = torch.mean(dice_coeff.mean(dim=0)[1:])
        return dice_coeff
  
        