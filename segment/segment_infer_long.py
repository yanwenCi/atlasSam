import torch
import numpy as np
import os
import json
from os.path import join
from torch.utils.data import DataLoader

from dataloader3d import dataset_loaders
from loss_fun import loss_fn, dice_coefficient
from monai.networks import nets 
import argparse
import logging
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import cv2
import nibabel as nib   

def testdataloader(path):
    t2_path =join(path, 't2w')
    t2_files = os.listdir(t2_path)
    t2_files = [join(t2_path,f) for f in t2_files if f.endswith('.nii.gz')]
    t2_files.sort()
    adc_files = [f.replace('t2w', 'adc') for f in t2_files]
    dwi_files = [f.replace('t2w', 'dwi') for f in t2_files]
    def nii_load_norm(path):
        nii = nib.load(path)
        img = nii.get_fdata()
        img = (img - img.min())/(img.max() - img.min())
        datalist = {'data': torch.from_numpy(img[np.newaxis, np.newaxis, ...]).float(),
                    'head': nii.header,
                    'affine': nii.affine,
                    'name': path.split('/')[-1]
                    }
        return datalist
    t2data_list = [nii_load_norm(t2_files[i]) for i in range(len(t2_files))] 
        # adcdata_list = [nib.load(adc_file[i]).get_fdata()]
        # dwidata_list = [nib.load(dwi_file[i]).get_fdata()]
    return t2data_list  

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


def setup_model(args) -> torch.nn.Module:
    """Initialize the model based on user-defined parameters."""
    model = nets.UNet(spatial_dims=3, in_channels=args.inch, out_channels=args.outch,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
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

def test(args):
    """Testing function."""
    device = setup_device(args.gpus)
    save_dir = join(args.save_dir, args.load_from_dir)
    model_path = join(save_dir, f'best.pth')

    model = setup_model(args)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.eval()
    device = setup_device(args.gpus)
    data_path = join(args.path)
 
    dice_scores = []
    y_true_lesions, y_pred_lesions = [], []

    test_dataloader = testdataloader(data_path)
    print('test data length', len(test_dataloader))
    dice_bin = 0
    save_output=True
    for i, inputs in enumerate(test_dataloader):
            # output is logits, do softmax before putting into dice function
        with torch.no_grad():
            data = inputs['data'].to(device)
           
            outputs = model(data)
            outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.argmax(dim=1)[0]
            # print(torch.unique(outputs), torch.unique(labels)) 
                
            if save_output:
                zind = outputs.shape[-1]//2
                concat_image = np.concatenate([255*np.rot90(data.cpu().numpy()[0,0,:,:,zind], k=3),
                                                   127*np.rot90(outputs.cpu().numpy()[:,:,zind], k=3)], axis=1)
                    
                __save = join(args.save_dir, 'masks')
                os.makedirs(__save, exist_ok=True)
                cv2.imwrite(join(__save, inputs['name'].replace('.nii.gz', '.png')), concat_image)
        
            outputs = nib.Nifti1Image(outputs.cpu().numpy(), affine=inputs['affine'], header=inputs['head'])
            nib.save(outputs, join(__save, inputs['name']))
    


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="3D UNet Prostate Segmentation")
    
    # General arguments
    parser.add_argument('--data_root', type=str, default='../../Dataset/ProstateDatasets/data_split_files', help='Path to data')
    parser.add_argument('--path', type=str, default='picai_zones_ratio0.7', help='Path to data')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train', help='Train or test phase')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'dice', 'focal'], default='dice',
                        help='Loss function')
    parser.add_argument('--inch', type=int, default=1, help='Number of input channels')
    parser.add_argument('--outch', type=int, default=3, help='Number of output channels')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--epoch_load', default=100, help='Epoch number for loading model')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every X epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--crop_size', type=int, default=None, help='Crop size')
    parser.add_argument('--model_type', type=str, default='unet3d', help='Model type')
    parser.add_argument('--gpus', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--continue_train', action='store_true', help='Resume training from a checkpoint')

    # Testing
    parser.add_argument('--load_from_dir', type=str, default='unet3d_picai_zones_ratio0.7', help='Directory to load model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    '''
    usage: test: python segment.py --path picai_zones_ratio0.7 --epoch_load best --phase test --load_from_dir unet3d_1-uclH-data_ratio0.7
           train: python segment.py --path 1-uclH-data_ratio0.8 --phase train
    '''
    
    if args.phase == 'test':
        test(args)
