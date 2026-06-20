# Implementations of loss functions and metrics that are useful for both estimation and evaluation
import torch


class ROILoss():
    def __init__(self, w_overlap=1.0, w_class=1.0, batch_wise=False) -> None:
        self.w_overlap = w_overlap
        self.w_class = w_class
        self.batch_wise = batch_wise

    def __call__(self, roi0, roi1):
        '''
        Implements Dice as the overlap loss cross all masks
        roi0: torch.tensor of shape (C,D1,H1,W1) for 3d where C is the number of masks
                                    (D1,H1,W1) for 2d
        roi1: torch.tensor of shape (C,D1,H1,W1) for 3d where C is the number of masks
                                    (D1,H1,W1) for 2d
        '''
        if self.batch_wise:
            roi0 = roi0.flatten()
            roi1 = roi1.flatten()
        else:
            roi0 = roi0.flatten(start_dim=1)
            roi1 = roi1.flatten(start_dim=1)
       
        loss = 0
        if self.w_overlap != 0:
            loss += self.w_overlap * self.overlap_loss(roi0, roi1)
        if self.w_class != 0:
            loss += self.w_class * self.class_loss(roi0, roi1)
        return loss

    def overlap_loss(self, roi0, roi1, eps=1e-8):
        '''
        Implements Dice as the overlap loss
        '''
        intersection = (roi0 * roi1).sum(dim=-1)
        union = roi0.sum(dim=-1) + roi1.sum(dim=-1)
        overlap = 2*intersection / (union+eps)
        return 1 - overlap.mean()
    
    def class_loss(self, roi0, roi1):
        '''
        Implements mean-square-error as the classification loss
        '''
        mse = ((roi0 - roi1)**2).mean(dim=-1)
        return mse.mean()


class DDFLoss():
    def __init__(self, type='l2grad') -> None:
        self.type = type

    def __call__(self, ddf):
        '''
        ddf: torch.tensor of shape (B,H,W,D,3) for 3D, or (B,H,W,2) for 2D
        '''
        if len(ddf.shape) == 5:  # 2D: (B,H,W,2)
            if self.type.lower() == "l2grad":
                loss = self.gradient_norm_2d(ddf, l1_flag=False)
            elif self.type.lower() == "l1grad":
                loss = self.gradient_norm_2d(ddf, l1_flag=True)
            elif self.type.lower() == "bending":
                loss = self.bending_energy_2d(ddf)
            else:
                raise ValueError(f"Unknown DDFLoss type: {self.type}")
        elif len(ddf.shape) == 6:  # 3D: (B,H,W,D,3)
            if self.type.lower() == "l2grad":
                loss = self.gradient_norm(ddf, l1_flag=False)
            elif self.type.lower() == "l1grad":
                loss = self.gradient_norm(ddf, l1_flag=True)
            elif self.type.lower() == "bending":
                loss = self.bending_energy(ddf)
            else:
                raise ValueError(f"Unknown DDFLoss type: {self.type}")
        else:
            raise ValueError("Unsupported DDF shape.")
        return loss

    ## 3D versions
    def gradient_norm(self, ddf, l1_flag=False):
        dFdx, dFdy, dFdz = self.ddf_gradients(ddf)
        if l1_flag:
            grad_norms = torch.abs(dFdx) + torch.abs(dFdy) + torch.abs(dFdz)
        else:
            grad_norms = dFdx**2 + dFdy**2 + dFdz**2
        return grad_norms.mean()

    def bending_energy(self, ddf):
        dFdx, dFdy, dFdz = self.ddf_gradients(ddf)
        d2Fdxx, d2Fdxy, d2Fdxz = self.ddf_gradients(dFdx)
        d2Fdyx, d2Fdyy, d2Fdyz = self.ddf_gradients(dFdy)
        d2Fdzx, d2Fdzy, d2Fdzz = self.ddf_gradients(dFdz)
        bending_energy = d2Fdxx**2 + d2Fdyy**2 + d2Fdzz**2 + \
                         2 * d2Fdxy * d2Fdyx + 2 * d2Fdxz * d2Fdzx + 2 * d2Fdyz * d2Fdzy
        return bending_energy.mean()

    @staticmethod
    def ddf_gradients(ddf):
        '''
        ddf: (B,H,W,D,3)
        Returns gradients: (B,H,W,D,3)
        '''
        dXdx, dXdy, dXdz = torch.gradient(ddf[..., 0], dim=(1, 2, 3))
        dYdx, dYdy, dYdz = torch.gradient(ddf[..., 1], dim=(1, 2, 3))
        dZdx, dZdy, dZdz = torch.gradient(ddf[..., 2], dim=(1, 2, 3))

        dFdx = torch.stack([dXdx, dYdx, dZdx], dim=-1)
        dFdy = torch.stack([dXdy, dYdy, dZdy], dim=-1)
        dFdz = torch.stack([dXdz, dYdz, dZdz], dim=-1)

        return dFdx, dFdy, dFdz

    ## 2D versions
    def gradient_norm_2d(self, ddf, l1_flag=False):
        dFdx, dFdy = self.ddf_gradients_2d(ddf)
        if l1_flag:
            grad_norms = torch.abs(dFdx) + torch.abs(dFdy)
        else:
            grad_norms = dFdx**2 + dFdy**2
        return grad_norms.mean()

    def bending_energy_2d(self, ddf):
        dFdx, dFdy = self.ddf_gradients_2d(ddf)
        d2Fdxx, d2Fdxy = self.ddf_gradients_2d(dFdx)
        d2Fdyx, d2Fdyy = self.ddf_gradients_2d(dFdy)
        bending_energy = d2Fdxx**2 + d2Fdyy**2 + 2 * d2Fdxy * d2Fdyx
        return bending_energy.mean()

    @staticmethod
    def ddf_gradients_2d(ddf):
        '''
        ddf: (B,H,W,2)
        Returns: dFdx, dFdy of shape (B,H,W,2)
        '''
        dXdx, dXdy = torch.gradient(ddf[..., 0], dim=(1, 2))
        dYdx, dYdy = torch.gradient(ddf[..., 1], dim=(1, 2))

        dFdx = torch.stack([dXdx, dYdx], dim=-1)
        dFdy = torch.stack([dXdy, dYdy], dim=-1)

        return dFdx, dFdy
