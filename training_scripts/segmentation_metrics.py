import torch
import numpy as np
import PIL

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
    
    

'''
convert an image outputted by the segmentation LDM to a segmentation mask
model_output: PIL.Image
returns: np.ndarray with pixels [0,0,0](bg) or [255,255,255](truth)
'''
def convert_seg_array_to_binary_rgb(model_output:PIL.Image):
    # replace all pixels in seg_arr that dont have 0,0 in their GB channels with 255,255,255
    seg_arr = np.asarray(model_output).copy()
    mask = (seg_arr[:,:,0] == 0) | (seg_arr[:,:,1] == 0) | (seg_arr[:,:,2] == 0)
    seg_arr[mask] = [255,255,255]
    # replace all pixels that arent [255,255,255 with [0,0,0]
    mask = (seg_arr != [255,255,255]).any(axis=2)
    new_seg_arr = seg_arr
    new_seg_arr[mask] = [0,0,0]
    
    return new_seg_arr

'''
helper function to convert RGB image to binary mask
seg_rgb_arr - numpy array of shape (h, w, 3), where the segmented object is white and the background is black
'''
def convert_RGB_to_seg_mask(seg_rgb_arr:np.ndarray):
    new_array = np.all(seg_rgb_arr == [255, 255, 255], axis=-1).astype(int)
    return new_array


'''
compute mean IoU score for a list of images
pred: list containing numpy arrays of shape (h, w)
target: list containing numpy arrays of shape (h, w)
'''
def miou_score(pred:list, target:list) -> float:
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

'''
model_outputs: list of PIL images
target_masks: list of numpy arrays of shape (h, w)
'''
def compute_mdice_miou_for_model_outputs(model_outputs:list, target_masks:list):
    # convert model outputs to binary masks
    model_outputs_binary_masks = []
    for model_output in model_outputs:
        rgb_segmask = convert_seg_array_to_binary_rgb(model_output) 
        bin_segmask = convert_RGB_to_seg_mask(rgb_segmask)
        model_outputs_binary_masks.append(bin_segmask)
    # compute mdice
    mdice = mdice_coeff(model_outputs_binary_masks, target_masks)
    miou = miou_score(model_outputs_binary_masks, target_masks)
    return mdice, miou