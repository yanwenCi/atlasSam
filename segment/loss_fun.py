
import torch
import torch.nn as nn
import torch.nn.functional as F
from  scipy.spatial import KDTree
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def loss_fn(loss_type):
    """
    Returns the loss function based on the loss type.

    Args:
        loss_type (str): Name of the loss function.

    Returns:
        loss_fn (nn.Module): Loss function.
    """
    if loss_type == 'cross_entropy':
        return CrossEntropyLoss3D()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss3D()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")




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
            pred (Tensor): Predicted tensor (B, C, D, H, W) with probabilities.
            target (Tensor): Ground truth tensor (B, C, D, H, W), one-hot encoded.
        
        Returns:
            dice_loss (Tensor): Computed dice loss.
        """
        # Flatten the tensors (B, C, D, H, W) -> (B, C, N)
        num_calss = pred.shape[1]
        # print(pred.max(), pred.min(), target.max(), target.min())
        pred = torch.softmax(pred, dim=1)
        target = F.one_hot(target.long(), num_classes=num_calss).permute(0, 4, 1, 2, 3)

        # Compute Dice coefficient
        intersection = torch.sum(pred * target, dim=[2,3,4])
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum(dim=[2,3,4]) + target.sum(dim=[2,3,4]) + self.smooth)
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow(pred.argmax(1)[0,...,48 ].cpu().numpy(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(target.argmax(1)[0,...,48].cpu().numpy(), cmap='gray')
        plt.savefig('debug.png')
        plt.close()

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


class evaluation_metrics():
    def __init__(self, classes, spacing=(1, 1, 1)):
        super(evaluation_metrics, self).__init__()
        self.classes = classes
        self.spacing = spacing
        

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
        target = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3)
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=2)
        dice_coeff = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)
        # dice_coeff = torch.mean(dice_coeff.mean(dim=0)[1:])
        return dice_coeff.mean(dim=0)
    
    def distance_map(self, seg, gt):
        """
        Compute Hausdorff Distance for each class in a 3D segmentation mask.
    
        Args:
        seg (numpy.ndarray): Predicted segmentation of shape (B, C, W, H, D).
        gt (numpy.ndarray): Ground truth mask of shape (B, W, H, D).
        
        Returns:
        numpy.ndarray: Hausdorff distances of shape (B, C).
        """
        B, C, W, H, D = seg.shape  # Segmentation shape
        out = torch.zeros((B, C))  # Store Hausdorff distances

        for b in range(B):  # Loop over batch
            for i in range(1, C):  # Loop over classes (skip background)
                seg_points = torch.argwhere(seg[b, i] > 0.5).cpu().numpy()  # Get voxel indices for segmentation
                gt_points = torch.argwhere(gt[b] == i).cpu().numpy()  # Get voxel indices for ground truth

                if seg_points.size == 0 or gt_points.size == 0:
                    out[b, i] = 100  # Large value if empty mask
                else:
                    seg_points = seg_points * np.array(self.spacing)  # Convert to physical coordinates
                    gt_points = gt_points * np.array(self.spacing)  # Convert to physical coordinates
                        # Build KD-Trees for fast nearest neighbor search
                    tree_seg = KDTree(seg_points)
                    tree_gt = KDTree(gt_points)

                    # Compute nearest neighbor distances
                    distances_seg_to_gt = tree_gt.query(seg_points)[0]  # Closest gt point to each seg point
                    distances_gt_to_seg = tree_seg.query(gt_points)[0]  # Closest seg point to each gt point

                    # Compute Mean Surface Distance (MSD)
                    surface_distance = np.concatenate([distances_seg_to_gt, distances_gt_to_seg])
                    out[b, i] = np.percentile(surface_distance, 95)#surface_distance.mean()
        return out.mean(0)  # Shape: (
            
    def ROC_multi(self, pred, gt):
        pred = pred.view(pred.shape[0], self.classes, -1)
        gt = gt.view(gt.shape[0], -1)
        metrics = {}
        for i in range(self.classes):
            pred_class = pred[:, i]
            gt_class = (gt == i).float()
            fpr, tpr, _ = roc_curve(gt_class, pred_class)
            roc_auc = auc(fpr, tpr)
            metrics[f'class_{i}'] = [fpr, tpr, roc_auc]
        return metrics

            
        