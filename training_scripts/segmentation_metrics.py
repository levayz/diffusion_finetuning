import torch
import numpy as np
from PIL import Image
from typing import List
import sys
sys.path.append('/disk4/Lev/Projects/diffusion_finetuning')
sys.path.append('/disk4/Lev/Projects/diffusion_finetuning/training_scripts')

from utils.utils import convert_seg_array_to_binary_rgb, convert_RGB_to_bin_seg_mask, convert_PIL_images_to_binary_masks

'''
pred - numpy array of shape (h, w)
target - numpy array of shape (h, w)
'''
def dice_coeff(pred:np.ndarray, target:np.ndarray) -> float:
    # Check that the inputs have the same shape
    assert pred.shape == target.shape
    
    # Convert the arrays to binary values
    pred_binary = (pred > 0).flatten()
    target_binary = (target > 0).flatten()
    
    # Compute the intersection and sum of the binary arrays
    intersection = np.sum(pred_binary * target_binary)
    sum_arrays = np.sum(pred_binary) + np.sum(target_binary)
    
    # Compute the Dice coefficient
    if sum_arrays == 0:
        return 1.0  # Special case when both arrays are empty
    else:
        dice = (2.0 * intersection) / sum_arrays
        return dice
                
'''
pred - numpy array of shape (h, w)
target - numpy array of shape (h, w)
'''                    
def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    # Check that the inputs have the same shape
    assert pred.shape == target.shape
    # check the pred has only 0 or 1 values
    assert len(np.unique(pred)) <= 2 and len(np.unique(target)) <= 2
    # Convert the arrays to binary values
    pred_binary = (pred > 0)
    target_binary = (target > 0)
    
    # Compute the intersection and union of the binary arrays
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    
    # Compute the IoU score
    if union == 0:
        return 1.0  # Special case when both arrays are empty
    else:
        iou = intersection / union
        return iou
    
def hd95(pred: np.ndarray, target:np.ndarray):
    # Check that the inputs have the same shape
    assert pred.shape == target.shape
    # check the pred has only 0 or 1 values
    assert len(np.unique(pred)) <= 2 and len(np.unique(target)) <= 2
    # Convert the arrays to binary values
    pred_binary = (pred > 0)
    target_binary = (target > 0)
    
    # Get the coordinates of the boundary points for each array
    gt_boundary = np.array(np.where(target_binary == 1)).T
    pred_boundary = np.array(np.where(pred_binary == 1)).T
    
    # Check that there are boundary points in both arrays
    if len(gt_boundary) == 0 or len(pred_boundary) == 0:
        return None
    
    # Calculate the distances between each pair of boundary points
    distances = np.sqrt(((gt_boundary[:, None, :] - pred_boundary[None, :, :]) ** 2).sum(axis=2))
    
    # Get the 95th percentile of the distances
    hd95 = np.percentile(distances, 95)
    
    return hd95
    
    

'''
convert an image outputted by the segmentation LDM to a segmentation mask
model_output: PIL.Image
returns: np.ndarray with pixels [0,0,0](bg) or [255,255,255](truth)
'''
def convert_seg_array_to_binary_rgb_med(model_output:Image.Image, threshold:int=0):
    # replace all pixels in seg_arr that dont have 0,0 in their GB channels with 255,255,255
    seg_arr = np.asarray(model_output).copy()
    mask = (seg_arr[:,:,0] <= threshold) | (seg_arr[:,:,1] <= threshold) | (seg_arr[:,:,2] <= threshold)
    seg_arr[mask] = [255,255,255]
    # replace all pixels that arent [255,255,255] with [0,0,0]
    mask = (seg_arr != [255,255,255]).any(axis=2)
    new_seg_arr = seg_arr
    new_seg_arr[mask] = [0,0,0]
    
    return new_seg_arr

'''
convert an image outputted by the segmentation LDM to a segmentation mask
assumes that the background is darker than the segmented object
model_output: PIL.Image
returns: np.ndarray with pixels [0,0,0](bg) or [255,255,255](truth)
'''
# def convert_seg_array_to_binary_rgb(model_output:Image.Image, up_threshold:int=0, down_threshold:int=0):
#     # replace all pixels in seg_arr that dont have 0,0 in their GB channels with 255,255,255
#     # replace very bright pixel or very dark pixels with 255,255,255
#     seg_arr = np.asarray(model_output).copy()
#     mask = (seg_arr[:,:,0] >= up_threshold) | (seg_arr[:,:,1] >= up_threshold) | (seg_arr[:,:,2] >= up_threshold) | \
#             ((seg_arr[:,:,0] <= down_threshold) & (seg_arr[:,:,1] <= down_threshold) & (seg_arr[:,:,2] <= down_threshold))
#     seg_arr[mask] = [255,255,255]
#     # replace all pixels that arent [255,255,255] with [0,0,0]
#     mask = (seg_arr != [255,255,255]).any(axis=2)
#     new_seg_arr = seg_arr
#     new_seg_arr[mask] = [0,0,0]
    
#     return new_seg_arr



# '''
# helper function to convert RGB image to binary mask
# seg_rgb_arr - numpy array of shape (h, w, 3), where the segmented object is white and the background is black
# '''
# def convert_RGB_to_bin_seg_mask(seg_rgb_arr:np.ndarray):
#     new_array = np.all(seg_rgb_arr == [255, 255, 255], axis=-1).astype(int)
#     return new_array


'''
compute mean IoU score for a list of images
pred: list containing numpy arrays of shape (h, w)
target: list containing numpy arrays of shape (h, w)
'''
def miou_score(pred: List[np.ndarray], target: List[np.ndarray]) -> float:
    # Check that the inputs have the same shape
    assert len(pred) == len(target)
    
    # Compute the IoU score for each image
    iou_scores = []
    for i in range(len(pred)):
        iou_scores.append(iou_score(pred[i], target[i]))
    
    # Compute the mean IoU score
    miou = np.mean(iou_scores)
    return miou

'''
compute mean dice coefficient for a list of images
pred: list containing numpy arrays of shape (h, w)
target: list containing numpy arrays of shape (h, w)
'''
def mdice_coeff(pred:list, target:list) -> float:
    # Check that the inputs have the same shape
    assert len(pred) == len(target)
    
    # Compute the Dice coefficient for each image
    dice_scores = []
    for i in range(len(pred)):
        dice_scores.append(dice_coeff(pred[i], target[i]))
    
    # Compute the mean Dice coefficient
    mdice = np.mean(dice_scores)
    return mdice

# def convert_PIL_images_to_binary_masks(model_outputs: List[Image.Image], up_thresh=None, down_thresh=None):
#     # convert model outputs to binary masks
#     model_outputs_binary_masks = []
#     for model_output in model_outputs:
#         if up_thresh is not None and down_thresh is not None:
#             rgb_segmask = convert_seg_array_to_binary_rgb(model_output, up_threshold=up_thresh, down_threshold=down_thresh)
#         else:
#             rgb_segmask = convert_seg_array_to_binary_rgb(model_output)
#         bin_segmask = convert_RGB_to_bin_seg_mask(rgb_segmask)
#         model_outputs_binary_masks.append(bin_segmask)
#     return model_outputs_binary_masks

def compute_mdice_miou_for_model_outputs(model_outputs: List[Image.Image], target_masks: List[np.ndarray]):
    model_outputs_binary_masks = convert_PIL_images_to_binary_masks(model_outputs)
    # compute mdice
    mdice = mdice_coeff(model_outputs_binary_masks, target_masks)
    miou = miou_score(model_outputs_binary_masks, target_masks)
    return mdice, miou

def APmask(prediction:np.ndarray, target:np.ndarray):
    # Check that the inputs have the same shape
    assert prediction.shape == target.shape
    # check the pred has only 0 or 1 values
    assert len(np.unique(prediction)) <= 2 and len(np.unique(target)) <= 2
    
    # Compute the true positives (TP), false positives (FP),
    # and false negatives (FN)
    pred_binary = (prediction > 0)
    target_binary = (target > 0)
    
    tp = np.sum(np.logical_and(pred_binary, target_binary))
    fp = np.sum(pred_binary) - tp
    fn = np.sum(target_binary) - tp

    if tp + fp == 0:
        return 0 , 0, 0
    
    # Compute the precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Compute the average precision
    ap = precision * recall

    return ap, precision, recall

def mAPmask(predictions: List[np.ndarray], targets: List[np.ndarray]):
    # Compute the average precision for each image
    aps = []
    for i in range(len(predictions)):
        aps.append(APmask(predictions[i], targets[i]))

    # Compute the mean average precision
    map = np.mean(aps)
    return map

def compute_mAPmask_for_model_outputs(model_outputs: List[Image.Image], target_masks: List[np.ndarray]):
    model_outputs_binary_masks = convert_PIL_images_to_binary_masks(model_outputs)
    map = mAPmask(model_outputs_binary_masks, target_masks)
    return map


def detect_object(prediction:np.ndarray, target:np.ndarray, iou_thresh=0.5):
    # compute the IoU score, positive if above threshold
    if iou_score(prediction, target) >= iou_thresh:
        return 1
    return 0

def mAPdetection(predictions: List[np.ndarray], targets: List[np.ndarray], miou_thresh=0.33):
    # Compute the average precision for each image
    aps = []
    for i in range(len(predictions)):
        aps.append(detect_object(predictions[i], targets[i], miou_thresh))

    # Compute the mean average precision
    map = np.mean(aps)
    return map

def compute_mAPdetection_for_model_outputs(model_outputs: List[Image.Image], target_masks: List[np.ndarray], miou_thresh=0.5):
    model_outputs_binary_masks = convert_PIL_images_to_binary_masks(model_outputs)
    map = mAPdetection(model_outputs_binary_masks, target_masks, miou_thresh)
    return map

from SegMapDataset import SegMapDataset
import matplotlib.pyplot as plt

def plot_seg_arr(in_img, seg_arr, prediction, origin_output):
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    ax[0].imshow(in_img)
    ax[1].imshow(seg_arr)
    ax[2].imshow(prediction)
    ax[3].imshow(origin_output)
    
    plt.show()

from training_scripts.pretrained_segmentation_head import detection_mask_rcnn

def test_model_segmentation(img_dir,
                            seg_dir,
                            annotaitons_dir,
                            pipe,
                            max_examples=float('inf'),
                            bin_segmask_threshold=180,
                            strength=0.1,
                            guidance_scale=12,
                            print_every=100):
    ds = SegMapDataset(img_dir, seg_dir, annotaitons_dir, 'segmentation map of', size=512, resize=True)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    # print('hello')
    dice_scores = []
    iou_scores = []
    ap_mask_scores = []
    ap_detect_scores = []
    hd95_scores = []
    for step, batch in enumerate(dataloader):
        if step - 1 > max_examples:
            break
        # in_img = batch['instance_images_clean'].numpy().squeeze()
        # in_img = Image.fromarray(in_img).convert('RGB').resize((512, 512))
        in_img = batch['instance_images_path'][0]
        in_img = Image.open(in_img).convert('RGB').resize((512, 512))
        
        seg_arr = batch['instance_segmap_images'].numpy().squeeze()
        
        prompt = 'segmentation map of ' + batch['instance_classes'][0]
        output = pipe(prompt, in_img, strength=strength, guidance_scale=guidance_scale, modified_unet=True, segmentation=True).images[0]
        origin_output = output.copy()
        output_arr = np.array(output)
        up_thresh = output_arr.max() * 0.75
        down_thresh = output_arr.max() * 0.1
        
        output = convert_PIL_images_to_binary_masks([output], up_thresh=up_thresh, down_thresh=down_thresh)[0]
        seg_arr = seg_arr[0].copy()
        seg_arr[seg_arr == 255] = 1
        
        dice_scores.append(dice_coeff(output, seg_arr))
        iou_scores.append(iou_score(output, seg_arr))
        ap_mask_scores.append(APmask(output, seg_arr))
        ap_detect_scores.append(detect_object(output, seg_arr))
        hd95_score = hd95(output, seg_arr)
        if hd95_score is not None:
            hd95_scores.append(hd95_score)
        
        if step % print_every == 0:
            print(f'mdice: {np.mean(dice_scores)}, miou: {np.mean(iou_scores)}, mAPmask: {np.mean(ap_mask_scores)}, mAPdetect: {np.mean(ap_detect_scores)}, mhd95: {np.mean(hd95_scores)}')
            print(prompt)
            plot_seg_arr(in_img ,seg_arr, output, origin_output)
            # detection_mask_rcnn(origin_output)
    
    scores = {'dice' : dice_scores,
              'iou': iou_scores,
              'ap_mask' : ap_mask_scores,
              'ap_detect': ap_detect_scores,
              'hd95': hd95_scores}
    return scores

from training_scripts.CocoStuffDataset import CocoStuffDataset
seen_classes = ['background',
                    'aeroplane',
                    'bicycle',
                    'bird',
                    'boat',
                    'bottle',
                    'bus',
                    'car',
                    'cat',
                    'chair',
                    'cow',
                    'dining table',
                    'dog',
                    'horse',
                    'motorbike',
                    'person',
                    'pottedplant',
                    'sheep',
                    'sofa',
                    'train',
                    'tv monitor']

def test_model_segmentation_on_unseen_classes(img_dir,
                                            seg_dir,
                                            coco_stuff_labels_file,
                                            image_list_path,
                                            pipe,
                                            seen_classes=seen_classes, 
                                            max_examples=float('inf'),
                                            bin_segmask_threshold=180,
                                            strength=0.1,
                                            guidance_scale=12,
                                            print_every=100):
    
    ds = CocoStuffDataset(img_dir,
                          seg_dir,
                          'segmentation map of',
                          coco_stuff_labels_file,
                          image_list_path=image_list_path,
                          size=512,
                          resize=True)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    # print('hello')
    dice_scores = []
    iou_scores = []
    ap_mask_scores = []
    ap_detect_scores = []
    step = 1
    for batch in dataloader:
        if step - 1 > max_examples:
            break
        instance_class = batch['instance_classes'][0]
        if instance_class in seen_classes:
            continue
        # in_img = batch['instance_images_clean'].numpy().squeeze()
        # in_img = Image.fromarray(in_img).convert('RGB').resize((512, 512))
        in_img = batch['instance_images_path'][0]
        in_img = Image.open(in_img).convert('RGB').resize((512, 512))
        
        seg_arr = batch['instance_segmap_images'].numpy().squeeze()
        
        prompt = 'segmentation map of ' + batch['instance_classes'][0]
        output = pipe(prompt, in_img, strength=strength, guidance_scale=guidance_scale, modified_unet=True, segmentation=True).images[0]
        origin_output = output.copy()
        output_arr = np.array(output)
        up_thresh = output_arr.max() * 0.75
        down_thresh = output_arr.max() * 0.1
    
        output = convert_PIL_images_to_binary_masks([output], up_thresh=up_thresh, down_thresh=down_thresh)[0]
        seg_arr = seg_arr[0].copy()
        seg_arr[seg_arr == 255] = 1
        
        dice_scores.append(dice_coeff(output, seg_arr))
        iou_scores.append(iou_score(output, seg_arr))
        ap_mask_scores.append(APmask(output, seg_arr))
        ap_detect_scores.append(detect_object(output, seg_arr))
        
        if step % print_every == 0:
            print(f'mdice: {np.mean(dice_scores)}, miou: {np.mean(iou_scores)}, mAPmask: {np.mean(ap_mask_scores)}, mAPdetect: {np.mean(ap_detect_scores)}')
            print(prompt)
            plot_seg_arr(in_img ,seg_arr, output, origin_output)
            # detection_mask_rcnn(origin_output)
        step += 1
    
    scores = {'dice:' : dice_scores,
              'iou': iou_scores,
              'ap_mask' : ap_mask_scores,
              'ap_detect': ap_detect_scores}
    return scores

from os import walk
import os
import SimpleITK as sitk
# add utils folder to path
from utils.utils import get_good_slices, get_good_slices_by_class
import random
from tqdm.auto import tqdm

def plot_4_seg_imgs(img1, img2, img3, img4):
    fig, ax = plt.subplots(1, 4, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[2].imshow(img3)
    ax[3].imshow(img4)
    plt.show()
    
def plot_seg_imgs(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()
    

def test_medical_segmentation_metrics(seg_path, vol_path, pipe, prompt,
                                      strength,
                                      guidance,
                                      dataset='lits',
                                      n_examples_per_ct=5,
                                      max_examples=100,
                                      print_every=10):
    assert dataset in ['lits', 'msd', 'miccai']
    # get list of all files in folder
    if dataset == 'lits':
        # this is for LITS17
        f = next(walk(seg_path), (None, None, []))[2]
        # take only files with 'segmentation' in them
        seg_filenames = [os.path.join(seg_path, filename) for filename in f if 'segmentation' in filename]
        # take only files with 'volume' in them
        vol_filenames = [os.path.join(vol_path, os.path.basename(filename).replace('segmentation', 'volume')) for filename in seg_filenames]
    elif dataset == 'msd':
        # this is for MSD
        f_seg = next(walk(seg_path), (None, None, []))[2]
        # take only files that have a corresponding volume file, the filename should be the same
        seg_filenames = [os.path.join(seg_path, filename) for filename in f_seg if not filename.startswith('.')]
        vol_filenames = [os.path.join(vol_path, os.path.basename(filename)) for filename in seg_filenames]
    elif dataset =='miccai':
        f_seg = next(walk(seg_path), (None, None, []))[2]
        # take only files that have a corresponding volume file, the filename should be the same
        seg_filenames = [os.path.join(seg_path, filename) for filename in f_seg if not filename.startswith('.')]
        vol_filenames = [os.path.join(vol_path, os.path.basename(filename).replace('label', 'img')) for filename in seg_filenames]
        
    dice_scores = []
    iou_scores = []
    ap_mask_scores = []
    hd95_scores = []
    
    global_step = -1
    
    progress_bar = tqdm(
        range(len(seg_filenames)))
    progress_bar.set_description("Patient")
    
    for i, item in enumerate(zip(seg_filenames, vol_filenames)):
        f_seg, f_vol = item
        # check if f has '.nii' in it
        if '.nii' not in f_seg or '.nii' not in f_vol:
            continue
        
        seg_img = sitk.ReadImage(f_seg)
        seg_arr = sitk.GetArrayFromImage(seg_img)
        img = sitk.ReadImage(f_vol)
        img_arr = sitk.GetArrayFromImage(img)
          
        slices = get_good_slices_by_class(seg_arr, seg_class=1) # spleen is 1 in miccai2015
        # slices = [100]
        indices = random.sample(slices, min(n_examples_per_ct, len(slices)))
        
        done = False
        for i in indices:
            global_step += 1
            if global_step > max_examples:
                done = True
                break
            ct_slice = img_arr[i]
            # remove all unnecessary classes from seg slice
            seg_slice = seg_arr[i]
            seg_slice[seg_slice != 1] = 0
            seg_slice = (seg_slice * 256).clip(0, 255) # make all values be 0 or 255

            in_img = Image.fromarray(ct_slice).convert('RGB')
            origin_output = pipe(prompt, in_img, strength=strength, guidance_scale=guidance, modified_unet=True, segmentation=True).images[0]
            output = convert_seg_array_to_binary_rgb(origin_output, 230, 0)           
            output = convert_RGB_to_bin_seg_mask(output)
            
            dice = dice_coeff(output, seg_slice)
            iou = iou_score(output, seg_slice)
            ap_mask, precision, recall = APmask(output, seg_slice)
            # hd95_score = hd95(output, seg_slice)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            ap_mask_scores.append(ap_mask)
            # if hd95_score is not None:
            #     hd95_scores.append(hd95_score)
                
            if global_step % print_every == 0:
                print(f'dice:{dice} iou:{iou} precision:{precision} recall:{recall}')
                print(f'global_step: {global_step}, \
                        mdice: {np.mean(dice_scores)}, \
                        miou: {np.mean(iou_scores)}, \
                        mAPmask: {np.mean(ap_mask_scores)}')
                print(f'slice:{i}')
                plot_4_seg_imgs(ct_slice, seg_slice, output, origin_output)
                # plot_seg_imgs(ct_slice, seg_slice)
                # plot_seg_imgs(origin_output, output)
        progress_bar.update(1)
            
        if done:
            break
                        
    scores = {'dice': dice_scores,
              'iou': iou_scores,
              'ap_mask': ap_mask_scores,
            #   'hd95': hd95_scores
              }
              
            
    return scores
