# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import nibabel as nib
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import sys
from dataloaders.dataloader3d import dataset_loaders
import argparse
import json
import os
from typing import Any, Dict, List
import numpy as np
import random

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    default="outputs",
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="vit_h",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

parser.add_argument(
    "--gpu", 
    type=int,
    default=0,
    help="The GPU to run generation on."
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)
amg_settings.add_argument(
    "--prompt-type",
    type=str,
    default='bbox',
    help="Prompt type to use for mask generation"
)

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def set_bbx(mask):
    mask = mask.astype(bool)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return np.array([x, y, w+x, h+y])[None,...]

def set_points(mask, number=1):
    mask = mask.astype(bool)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    [c1, c2] = [x+w//2, y+h//2]
    if number==1:
        return np.array([[c1, c2]])
    else:
        points = []
        points.append([h, w])
        points.append([random.choices(range(x, x+w), number-1), random.choices(range(y, y+h)), number-1])
        print(points)
        return np.array(points)
    
def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target)
    return 2.0 * intersection.sum() / (pred.sum() + target.sum())

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    dice_sum = 0
    
    # generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    predictor = SamPredictor(sam)
    
    test_dataset = dataset_loaders(path=args.input, phase='test', batch_size=1, np_var='vol', add_feat_axis=True)
    os.makedirs(args.output, exist_ok=True)
    fid=open(os.path.join(args.output, 'dice.txt'), 'w')
    atlas = test_dataset[0]['seg']
    for t in test_dataset:
        
        batch_target, tgt_seg = t['img'], t['seg']
        pred_mask = np.zeros(shape=tgt_seg.shape[:-1])
        
        base = os.path.basename(t['key'])
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)
        for i in range(10, batch_target.shape[0]-10):
            image=batch_target[i]
            if np.sum(tgt_seg[i]) <10:
                continue
            atls_bbx = set_bbx(atlas[i]) if args.prompt_type=='bbox' else None
            atls_points  = set_points(tgt_seg[i], number=10) if args.prompt_type=='point' else None
            # masks = generator.generate(image)
            # print(masks.shape)
            predictor.set_image(image)
            masks, _, _ = predictor.predict(point_coords=atls_points,
                                      point_labels=None,
                                      box=atls_bbx,
                                      mask_input=atlas[i],
                                      multimask_output=True)
            masks = masks.astype(int).transpose(1,2,0)
            pred_mask[i] = masks[:,:,0]
            # print(src_seg.shape, image.shape, masks.shape, masks.max())
            # print(np.unique(masks))
                # # write_masks_to_folder(masks, save_base)
                # import matplotlib.pyplot as plt
                # plt.imshow(masks)
                # plt.savefig(save_base+'.png')
            if i%10==0:
                cv2.imwrite(os.path.join(save_base, f'slice{i}_mask.png'), masks*255)
                cv2.imwrite(os.path.join(save_base, f'slice{i}_image.png'), image)
        dice = dice_score(pred_mask, tgt_seg[...,0])
        dice_sum += dice
        print(dice)
        fid.write(f'{base}: {dice}\n')
        nib.save(nib.Nifti1Image(pred_mask, np.eye(4)), os.path.join(args.output, base+'.nii.gz'))
        print("Done!")
    print('mean dice:', dice_sum/len(test_dataset))
    fid.write(f'Mean Dice score: {dice_sum/len(test_dataset)}')


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)
