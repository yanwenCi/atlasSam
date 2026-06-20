import torch
import numpy as np
import os
import json
from os.path import join
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader3d import dataset_loaders
from loss_fun import loss_fn, DiceLoss, bending_energy
from monai.networks import nets 
import argparse
import logging
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import nibabel as nib
import torch.nn.functional as F
import cv2
# from monai.networks.nets import VoxelMorph
from monai.networks.blocks import Warp
from networks import Voxelmorph
from AtlasNet import *
from torch.cuda.amp import autocast, GradScaler


img = nib.load('/raid/candi/Wen/Dataset/ProstateDatasets/3-picai-data/data-ROI-192-96/P-10000/t2.nii.gz')
affine_m = img.affine

def setup_device(gpu_id: int) -> str:
    """Setup computing device (GPU/CPU)."""
    return f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'


def get_dataloader(path: str, phase: str, batch_size: int, crop_size=None, istest=False) -> DataLoader:
    """Return a DataLoader for training or testing."""
    dataset = dataset_loaders(path, 
                 phase, batch_size=batch_size, 
                 np_var='vol', add_batch_axis=True, pad_shape=None,
                resize_factor=None, crop_size=crop_size, istest=istest, 
                transform=None, ifbin=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def setup_model(args, ) -> torch.nn.Module:
    # from atlasnet import Atlas2ImageRegModel, RegLossWeights
    # model = Atlas2ImageRegModel(loss_w=RegLossWeights(1e-3, 1e-4, 1e-3))
    # return model.to(setup_device(args.gpus))
    model = RegLite3D()
    return model.to(setup_device(args.gpus))

# def setup_model(args) -> torch.nn.Module:
#     """Initialize the model based on user-defined parameters."""
#     backbone = nets.VoxelMorphUNet(
#     spatial_dims=3,
#     in_channels=args.inch*2,
#     unet_out_channels=32,
#     dropout=0.1,
#     # norm="batch",
#     channels=(16, 32, 32, 32, 32, 32), final_conv_channels=(16, 16)
#     )

#     #    Then, a full VoxelMorph network is constructed using the specified backbone network.
#     model = nets.VoxelMorph(
#     backbone=backbone,
#     integration_steps=0,
#     half_res=True
#     )
#     print(model)
#     return model.to(setup_device(args.gpus))


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

def inverse_ddf(ddf, num_iters=20):
    """
    Compute inverse of a dense displacement field (DDF) using fixed-point iterations.

    Args:
        ddf (torch.Tensor): Displacement field of shape (1, 3, D, H, W).
        num_iters (int): Number of iterations for inversion.
        
    Returns:
        torch.Tensor: Inverse displacement field of the same shape as `ddf`.
    """
    device = ddf.device
    shape = ddf.shape[2:]  # Get (D, H, W)

    # Create initial inverse as negative of the original displacement field
    inv_ddf = -ddf.clone()

    # Create a regular grid normalized to [-1, 1]
    grid = torch.stack(torch.meshgrid(
        [torch.linspace(-1, 1, s, device=device) for s in shape], 
        indexing="ij"
    ), dim=0)  # Shape: (3, D, H, W)

    grid = grid.unsqueeze(0)  # Add batch dimension → (1, 3, D, H, W)

    for _ in range(num_iters):
        # Compute sampling locations: grid + current inverse displacement
        sample_grid = (grid + inv_ddf).permute(0, 2, 3, 4, 1)  # (1, D, H, W, 3)

        # Sample the original DDF at these locations
        inv_warped = F.grid_sample(ddf, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Update inverse displacement field
        inv_ddf = -(inv_warped + inv_ddf)

    return inv_ddf

def get_reference_grid3d(img, grid_size=None):
    '''
    return a 5d tensor of the grid, e.g.
    img --> (b, 1, h, w, z)
    out --> (b, 3, h, w, z)

    if grid_size is not None, then return a 3d grid with the size of grid_size
    grid_size --> (gh, gw, gz)
    '''
    if len(img.shape) > 3:
        batch = img.shape[0]
    else: 
        batch = 1
    
    shape = img.shape[-3:]
    
    if grid_size is not None:
        assert len(grid_size) == 3, "maybe not a 3d grid"
        shape = grid_size

    mesh_points = [torch.linspace(-1, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*mesh_points, indexing='ij'))  # shape:[3, x, y, z]
    grid = torch.stack([grid]*batch)  # add batch
    grid = grid.type(torch.FloatTensor) # [batch, 3, x, y, z]
    return grid

# def warp3d(img, ddf, ref_grid=None):
#     """
#     img: [batch, c, x, y, z]
#     new_grid: [batch, x, y, z, 3]
#     """

#     if ref_grid is None:
#         assert img.shape[-3:] == ddf.shape[-3:], "Shapes not consistent btw img and ddf."
#         grid = get_reference_grid3d(img).to(ddf.device)
#     else:
#         grid = ref_grid
  
#     new_grid = grid + ddf  # [batch, 3, x, y, z]
#     # print(new_grid.max(), new_grid.min(), grid.max(), grid.min(), ddf.max(), ddf.min())
#     new_grid = new_grid.permute(0, 2, 3, 4, 1)
#     new_grid = new_grid[..., [2, 1, 0]]
#     return F.grid_sample(img, new_grid, mode='bilinear', align_corners=False)


    
def atlas_input(batch_size, atlas, mask_atlas, device):
    """Prepare the atlas and mask atlas for input to the model."""
    #atlas shape: D, H, W; mask_atlas shape: D, H, W
    atlas = np.expand_dims(atlas, axis=[0,1]).repeat(batch_size, axis=0)
    mask_atlas = np.expand_dims(mask_atlas, axis=0).repeat(batch_size, axis=0)
    atlas = torch.tensor(atlas).repeat(args.inch, 1, 1, 1, 1,).to(device)#B, C, D, H, W
    mask_atlas = torch.tensor(mask_atlas).repeat(args.inch, 1, 1, 1,).to(device)#B,D,H,W
    return atlas, mask_atlas

   

def train(args):
    """Training function."""
    device = setup_device(args.gpus)
    save_dir = join(args.save_dir, f'{args.model_type}_{args.path}')
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=join(save_dir, args.log_dir))  # TensorBoard logger
    data_path = join(args.data_root, args.path)
    train_dataloader = get_dataloader(data_path, 'train', args.batch_size, args.crop_size)
    # atlas = np.load('atlas/atlas.npy')
    # mask_atlas = np.load('atlas/mode_mask.npy') 
    
    save_config(args, save_dir)
    logger = setup_logger(save_dir)

    # Model setup
    model = setup_model(args, )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device=='cuda'))
    # Load pretrained weights if resuming training
    if args.continue_train:
        model_path = join(save_dir, f'epoch_{args.epoch_load}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))

    atlas, mask_atlas = np.load('atlas/atlas4.npz')['atlas'], np.load('atlas/atlas4.npz')['mask_atlas']
    atlas_pz = mask_atlas==2
    atlas_cg = mask_atlas==1
    atlas = atlas.astype(np.float32)
    atlas_PZ = torch.from_numpy(atlas_pz.astype(np.float32)).to(device)
    atlas_CG = torch.from_numpy(atlas_cg.astype(np.float32)).to(device)
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        best_acc = 10
        imgs, msks = [], []
        move, mlabels = train_dataloader.dataset[0]['img0'], train_dataloader.dataset[0]['seg0']
       
        for i, data in enumerate(train_dataloader):
            B,_, D, H, W = data['img0'].shape
            move, mlabels = data['img0'].to(device), data['seg0'].to(device)
            atlas = np.expand_dims(atlas, axis=[0,1]).repeat(move.size(0), axis=0)
            atlas = torch.from_numpy(atlas).to(device)
            atlas_probs = {'CG': atlas_CG.repeat(move.size(0),1,1,1,1),
                           'PZ': atlas_PZ.repeat(move.size(0),1,1,1,1)}
            optimizer.zero_grad()
            out = model(atlas.float(), move.float(), atlas_probs=atlas_probs)
            priors = out['warped_probs'] 
            loss = out['loss']
            g_inv = inv_grid_total(out['A'], out['b'], out['v'], move.shape, steps=model.steps)

            I_back = F.grid_sample(move, g_inv, mode='bilinear', padding_mode='border', align_corners=True)
            Y = F.one_hot(mlabels.long().squeeze(1), num_classes=3).permute(0,4,1,2,3).float()  # [B,3,D,H,W]
            Y_back = F.grid_sample(Y, g_inv, mode='nearest', padding_mode='zeros', align_corners=True)

            s_i = model.lncc(atlas, I_back)   # scalar
            u_i = (jac_det(out['grid_total']) > 0).float().mean()
            w_i = (s_i.detach() * u_i.detach()).clamp_min(1e-3)

            scaler.scale(loss).backward()   # <-- THIS is where you backprop
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            # _, _ = validate(args, model, atlas, mask_atlas)
            if i % args.print_freq == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
                logger.info(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')
                zlice = 48
                input_show = [move[..., zlice].cpu(), atlas[..., zlice].cpu()]
                for c in range(2):
                    writer.add_images(f'Input{c}', input_show[c].repeat(1,3,1,1), epoch * len(train_dataloader) + i,
                                  dataformats='NCHW')
                for c in range(3):
                    writer.add_images('Output{c}', g_inv[:,c,...].cpu().detach().numpy()[:,None,:,:, zlice],
                                  epoch * len(train_dataloader) + i, dataformats='NCHW')
                
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

        #generate_new_atlas(imgs, msks)
        if (epoch + 1) % (args.save_freq) == 0:
            if (epoch+1)%(args.save_freq*5)==0:
                atlas, mask_atlas = generate_new_atlas(imgs, msks)
                cat_img = np.concatenate([atlas[...,46]*255, mask_atlas[...,46]*127+1], axis=1)
                cv2.imwrite(f'atlas/atlas{epoch}.png', cat_img)
                # print(mask_atlas.max(), mask_atlas.min())
                np.savez(f'atlas/atlas{epoch}.npz', atlas=atlas, mask_atlas=mask_atlas)
              
                nib.save(nib.Nifti1Image(atlas, affine_m), f'atlas/atlas{epoch}.nii.gz')
                nib.save(nib.Nifti1Image(mask_atlas, affine_m), f'atlas/mask{epoch}.nii.gz')
                print(f'atlas saved at epoch {epoch}')
                test_dataloader = get_dataloader(data_path, 'test', 1, args.crop_size, istest=True)
                atlas, atlas_CG, atlas_PZ, dice = evaluate_registration(model, test_dataloader, atlas, atlas_CG, atlas_PZ, device)
            
            
            if dice.mean() < best_acc:
                best_acc = dice.mean()
                torch.save(model.state_dict(), join(save_dir, f'best.pth'))
                
                logger.info(f'Best model saved at epoch {epoch}')
            logger.info(f'Dice Score: {dice.mean()}, Std: {std}')
            
    torch.save(model.state_dict(), join(save_dir, f'epoch_{epoch}.pth'))

    writer.close()

def generate_new_atlas(imgs, msks):
    """Generate a new atlas from the training data."""
    atlas = np.mean(np.stack(imgs, axis=0), axis=0).astype(np.float32) 
    mask_atlas = np.stack(msks, axis=0)
    mask_atlas, _ = stats.mode(mask_atlas, axis=0, keepdims=False)
    return atlas, mask_atlas.astype(np.float32) 

def evaluate_registration(reg, loader, psi_t2, pi_cg, pi_pz, device, max_iters=5):
    reg.eval()
    with torch.no_grad():
        sum_w = 0.0
        sum_I = torch.zeros_like(psi_t2)
        sum_CG = torch.zeros_like(pi_cg)
        sum_PZ = torch.zeros_like(pi_pz)
        sum_dice = 0.0
        for batch in loader:
            I = batch['image'].to(device)        # (B,1,D,H,W)
            M = batch['label'].to(device).long() # (B,1,D,H,W) integers {0,1,2}

            # forward atlas->image to get A,b,v  (atlas_t2 must match batch size)
            atlas_t2_b = psi_t2.repeat(I.size(0),1,1,1,1)
            out = reg(atlas_t2_b, I, atlas_probs=None)

            # inverse warp: image->atlas grid
            g_inv = inv_grid_total(out['A'], out['b'], out['v'], I.shape, steps=reg.steps)

            # warp image back
            I_back = F.grid_sample(I, g_inv, mode='bilinear', padding_mode='border', align_corners=True)

            # warp labels back as one-hot with nearest
            Y = torch.nn.functional.one_hot(M.squeeze(1), num_classes=3).permute(0,4,1,2,3).float()  # (B,3,D,H,W)
            Y_back = F.grid_sample(Y, g_inv, mode='nearest', padding_mode='zeros', align_corners=True)  # (B,3,...)
            Y_back = Y_back.clamp(0,1)
            dice = dice_multiclass_hard(Y_back, torch.cat([pi_cg, pi_pz], dim=1).repeat(I.size(0),1,1,1,1))
            sum_dice += dice.sum().item()
            # simple quality weights (LNCC to current atlas + topology fraction)
            s_i = reg.lncc(psi_t2.repeat(I.size(0),1,1,1,1), I_back)                 # scalar tensor
            u_i = ( (out['grid_total'].det() if hasattr(out['grid_total'], 'det') else
                      ( ( ( (out['grid_total'][:,:,:,1:,:] - out['grid_total'][:,:,:,:-1,:]).abs().mean() ) > 0 ) ) ) )
            # the above is a placeholder; use your jac_det function as in RegLite3D to compute pos_jac fraction:
            # u_i = (jac_det(out['grid_total']) > 0).float().mean()

            w = (s_i.detach())   # * u_i if you also compute pos-Jac fraction here
            w = w.view(-1,1,1,1,1)

            sum_w  += w.sum()
            sum_I  += (w * I_back).sum(dim=0, keepdim=True)
            sum_CG += (w * Y_back[:,1:2]).sum(dim=0, keepdim=True)
            sum_PZ += (w * Y_back[:,2:3]).sum(dim=0, keepdim=True)

        # recompute atlases (weighted means). For intensity you can use trimmed mean/median if you want extra robustness.
        psi_t2 = (sum_I / (sum_w + 1e-6)).clamp_min(0).to(device)
        pi_cg  = (sum_CG / (sum_w + 1e-6)).clamp(0,1).to(device)
        pi_pz  = (sum_PZ / (sum_w + 1e-6)).clamp(0,1).to(device)

        psi_t2.requires_grad_(False)
        pi_cg.requires_grad_(False)
        pi_pz.requires_grad_(False)
        
    return psi_t2, pi_cg, pi_pz, sum_dice / len(loader.dataset)

    # (optional) check convergence on sharpness change; break if small
    # sharp_prev, sharp_new = ..., ...
    # if (sharp_new - sharp_prev)/sharp_prev < 0.005: break


def validate(args, model, atlas, mask_atlas, save_output: bool = False, istest=False):
    """Validation function to compute Dice score."""
    model.eval()
    device = setup_device(args.gpus)
    data_path = join(args.data_root, args.path)
    # atlas, mask_atlas = np.load('atlas/atlas.npy'), np.load('atlas/mode_mask.npy')
    # atlas = np.expand_dims(atlas, axis=[0,1]).repeat(args.batch_size, axis=0)
    # mask_atlas = np.expand_dims(mask_atlas, axis=0).repeat(args.batch_size, axis=0)
    dice_scores = []
    dice_loss = DiceLoss()
    warp3d = Warp(mode='bilinear')
    y_true_lesions, y_pred_lesions = [], []
    if istest:
        phase = 'test'
        save_dir = join(args.save_dir, f'{args.load_from_dir}')
        dice_path = join(save_dir,'dice_scores_test.txt')
    else:
        phase = 'val'
        dice_path = join(args.save_dir, f'{args.model_type}_{args.path}','dice_scores.txt')
    test_dataloader = get_dataloader(data_path, phase, 1, args.crop_size, istest=True)

   
    with open(dice_path, 'w') as fid:
        for i, data in enumerate(test_dataloader):
            move, mlabels = data['img0'].to(device), data['seg0'].to(device)
            fix, flabels = atlas_input(move.shape[0], atlas, mask_atlas, device)
            # output is logits, do softmax before putting into dice function
            with torch.no_grad():
                # move_cat = torch.cat([move, mlabels.unsqueeze(1)], dim=1)
                # fix_cat = torch.cat([fix, flabels.unsqueeze(1)], dim=1)
                _, ddf = model(move.float(), fix.float())
                # sim = loss_fn(args.loss_type)(warped, move).cpu().numpy()
                warped = warp3d(move, ddf)
                mlabels = mlabels[:, None, ...]
                warped_seg = warp3d(mlabels, ddf)
                # inv_ddf = inverse_ddf(ddf)
                # warped_inv_seg = warp3d(flabels.unsqueeze(1).float(), inv_ddf)
                # print(warped_seg.max(), warped_seg.min(), flabels.max(), flabels.min())
                dice = dice_loss(warped_seg, flabels[:,None,...]).cpu().numpy()
                # inv_dice = dice_loss(warped_inv_seg[:,0], mlabels.squeeze(1)).cpu().numpy()
                dice_scores.append([ 1-dice])
               
                if save_output:
                    if not os.path.exists(f'outputs/{args.load_from_dir}'):
                        os.makedirs(f'outputs/{args.load_from_dir}', exist_ok=True)
                    fid.write(f'{dice}\n')
                    nib.save(nib.Nifti1Image(warped_seg[0,0].cpu().detach().numpy(), affine_m), f'outputs/{args.load_from_dir}/{i}_seg.nii.gz')
    dice_scores = np.array(dice_scores)
    return np.mean(dice_scores, axis=0), np.std(dice_scores, axis=0)


def test(args):
    """Testing function."""
    device = setup_device(args.gpus)
    save_dir = join(args.save_dir, args.load_from_dir)
    model_path = join(save_dir, f'{args.epoch_load}.pth')

    model = setup_model(args)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    atlas_ = np.load('atlas/atlas9.npz')
    atlas=atlas_['atlas']
    mask_atlas = atlas_['mask_atlas']
    dice, std = validate(args, model, atlas=atlas, mask_atlas=mask_atlas, save_output=True, istest=True)
    print(f'Dice Score: {dice}, Std: {std}')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="3D UNet Prostate Segmentation")
    
    # General arguments
    parser.add_argument('--data_root', type=str, default='../../Dataset/ProstateDatasets/data_split_files', help='Path to data')
    parser.add_argument('--path', type=str, default='picai_zones_ratio0.7', help='Path to data')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train', help='Train or test phase')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--loss_type', type=str, choices=['ce', 'dice', 'focal', 'ncc','mse'], default='mse',
                        help='Loss function')
    parser.add_argument('--noseg', action='store_true',  help='Segmentation task')
    parser.add_argument('--inch', type=int, default=1, help='Number of input channels')
    parser.add_argument('--outch', type=int, default=3, help='Number of output channels')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--epoch_load', default=100, help='Epoch number for loading model')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='Save model every X epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--crop_size', type=int, default=None, help='Crop size')
    parser.add_argument('--model_type', type=str, default='inverse', help='Model type')
    parser.add_argument('--gpus', type=int, default=0, help='GPU IDs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--continue_train', action='store_true', help='Resume training from a checkpoint')

    # Testing
    parser.add_argument('--load_from_dir', type=str, default='checkpoints', help='Directory to load model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
