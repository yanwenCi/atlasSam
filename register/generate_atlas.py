import dataloader3d as dl
import numpy as np
import os
from scipy import stats
import nibabel as nib

def load_data(path, phase):
    """Load data from a dataset."""
    dataset = dl.dataset_loaders(path, 
                 phase, batch_size=1, 
                 np_var='vol', add_batch_axis=False, )
    
    return dataset

path = '/raid/candi/Wen/Dataset/ProstateDatasets/data_split_files/picai_zones_ratio0.7'
dataset = load_data(path, 'train')
atlas, masks = [], []
for i, data in enumerate(dataset):

    atlas.append(data['img0'])
    masks.append(data['seg0'])

atlas = np.stack(atlas, axis=0).mean(axis=0)
print(f"Atlas shape: {atlas.shape}, atlas mean: {atlas.mean()}")    
masks = np.stack(masks, axis=0)
# Compute the mode along the first axis (N), which gives the most frequent pixel value at each voxel
mode_mask, _ = stats.mode(masks, axis=0, keepdims=False)
print(f"Mode mask shape: {mode_mask.shape}, mode mask mean: {np.unique(mode_mask)}")

atals = atlas.astype(float)
mode_mask = mode_mask.astype(np.uint8)
# Save the atlas and mode mask
np.save('atlas/atlas.npy', atlas)
np.save('atlas/mode_mask.npy', mode_mask)
img = nib.load('/raid/candi/Wen/Dataset/ProstateDatasets/3-picai-data/data-ROI-192-96/P-10000/t2.nii.gz')
affine = img.affine
nib.save(nib.Nifti1Image(atlas, affine), 'atlas/atlas.nii.gz')
nib.save(nib.Nifti1Image(mode_mask, affine), 'atlas/mod_mask.nii.gz')