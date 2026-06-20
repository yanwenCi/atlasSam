import torch
import numpy as np
import os
import json
from os.path import join
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader3d import dataset_loaders
from loss_fun import loss_fn
from loss_fun import evaluation_metrics as eval_metrics
from monai.networks import nets 
import argparse
import logging
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import cv2
import json
import nibabel as nib
import torch.nn.functional as F


def setup_device(gpu_id: int) -> str:
    """Setup computing device (GPU/CPU)."""
    return f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'


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
    save_dir = join(args.save_dir, f'{args.model_type}_{args.path}')
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=join(save_dir, args.log_dir))  # TensorBoard logger
    data_path = join(args.data_root, args.path)
    print('data path is ', data_path)
    train_dataloader = get_dataloader(data_path, 'train', args.batch_size, args.crop_size, args.spacing)

    save_config(args, save_dir)
    logger = setup_logger(save_dir)

    # Model setup
    model = setup_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = loss_fn(args.loss_type)
    
    # Load pretrained weights if resuming training
    if args.continue_train:
        model_path = join(save_dir, f'epoch_{args.epoch_load}.pth')
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
                logger.info(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')

                zlice = np.median(np.where(labels.detach().cpu() > 0)[-1]).astype(int) if labels.sum() > 0 else 48
                input_show = inputs[..., zlice].cpu()
                for c in range(args.inch):
                    writer.add_images(f'Input{c}', input_show[:,c,:,:][:,None,:,:].repeat(1,3,1,1), epoch * len(train_dataloader) + i,
                                  dataformats='NCHW')
                
                writer.add_images('Output', outputs.argmax(dim=1, keepdim=True).cpu().repeat(1, 3, 1, 1, 1)[..., zlice],
                                  epoch * len(train_dataloader) + i, dataformats='NCHW')
                writer.add_images('Ground Truth', labels.cpu()[:, None, ...].repeat(1, 3, 1, 1, 1)[..., zlice],
                                  epoch * len(train_dataloader) + i, dataformats='NCHW')

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

        if (epoch + 1) % args.save_freq == 0:
            dice, std = validate(args, model)
            # if np.mean(dice[1:]) > best_acc:
            if dice[1] > best_acc:
                best_acc = np.mean(dice[1:])
                torch.save(model.state_dict(), join(save_dir, f'best.pth'))
                logger.info(f'Best model saved at epoch {epoch}')
            logger.info(f'Dice Score: {dice}, Std: {std}')
    torch.save(model.state_dict(), join(save_dir, f'epoch_{epoch}.pth'))

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
    
    em = eval_metrics(classes=args.outch, spacing= args.spacing)
    if istest:
        phase = 'test'
        dice_path = join(args.save_dir, f'{args.load_from_dir}',f'{args.path}_eval_metrics.json')
    else:
        phase = 'val'
        dice_path = join(args.save_dir, f'{args.model_type}_{args.path}','eval_metrics.json')
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
                print(f"dice: {pro_dice}")
                dist = em.distance_map(outputs, labels).cpu().numpy()

                test_metrics[f'Image_{i}'] = {'Dice': dice[1:], 'Prostate Dice': pro_dice[1:], 'Distance': dist}
                metrics_all.append(list(dice[1:]) + list(pro_dice[1:]) + list(dist[1:]))
                
                # print(f'Dice Score: {dice}')
                if save_output:
                    
                    concat_image = np.concatenate([255*np.rot90(inputs.cpu().numpy()[0,0,:,:,48], k=3),
                                                   127*np.rot90(outputs_show.cpu().numpy()[0,:,:,48], k=3),
                                                     127*np.rot90(labels.cpu().numpy()[0,:,:,48], k=3)], axis=1)
                    
                    __save = join(args.save_dir, f'{args.load_from_dir}/{args.path}_visual')
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
    model_path = join(save_dir, f'best.pth')

    model = setup_model(args)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    args.batch_size = 1
    dice, std = validate(args, model, save_output=True, istest=True)
    print(f'Dice Score: {dice}, Std: {std}')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="3D UNet Prostate Segmentation")
    
    # General arguments
    parser.add_argument('--data_root', type=str, default='../../Dataset/ProstateDatasets/data_split_files', help='Path to data')
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
    # Testing
    parser.add_argument('--load_from_dir', type=str, default='unet3d_picai_zones_ratio0.7', help='Directory to load model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    '''
    usage: test: python segment.py --path picai_zones_ratio0.7 --epoch_load best --phase test --load_from_dir unet3d_1-uclH-data_ratio0.7
           train: python segment.py --path 1-uclH-data_ratio0.8 --phase train
    '''
    prompt = np.load('../register/atlas/atlas4.npz')['mask_atlas']
    print('prompt',prompt.shape)
    prompt = torch.from_numpy(prompt).unsqueeze(0).repeat(args.batch_size,1,1,1).to(setup_device(args.gpus))

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
