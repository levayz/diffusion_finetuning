import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def binary_segmentation_unet(image):
    # Load the image and the U-Net model
    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=3, out_channels=1, init_features=32, pretrained=True)

    # Define the transformation to apply to the input image
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Apply the transformation to the input image
    image = transform(image).unsqueeze(0)

    # Set the model to evaluation mode and make predictions on the input image
    unet_model.eval()
    with torch.no_grad():
        output = unet_model(image)

    # Convert the output tensor to a numpy array and threshold the values to generate a binary mask
    output = output.squeeze().cpu().numpy()
    binary_mask = np.zeros_like(output)
    binary_mask[output > 0.5] = 1

    # Convert the binary mask to a PIL image and return it
    binary_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
    return binary_mask


from torchvision import transforms
import torchvision.models as models

def binary_segmentation_maskrcnn(image):
    # Load the image and the Mask R-CNN model
    mask_rcnn_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Define the transformation to apply to the input image
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply the transformation to the input image
    image = transform(image)

    # Set the model to evaluation mode and make predictions on the input image
    mask_rcnn_model.eval()
    with torch.no_grad():
        predictions = mask_rcnn_model([image])[0]

    # Extract the binary mask from the predicted segmentation mask
    masks = predictions['masks'].cpu().numpy()
    binary_mask = np.zeros_like(masks[0, 0])
    for mask in masks:
        binary_mask = np.logical_or(binary_mask, mask[0])

    # Convert the binary mask to a PIL image and return it
    binary_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
    return binary_mask

import torchvision.models as models

def binary_segmentation_deeplab(image):
    # Load the image and the DeepLabv3+ model
    deeplab_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    deeplab_model.eval()
    
    # Define the transformation to apply to the input image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformation to the input image
    image = transform(image).unsqueeze(0)

    # make predictions on the input image
    with torch.no_grad():
        output = deeplab_model(image)['out']
    output_predictions = output.argmax(0)

    # plot_segmentation(output_predictions)
    return output
    # Convert the output tensor to a numpy array and threshold the values to generate a binary mask
    output = output.squeeze().cpu().numpy()
    binary_mask = np.zeros_like(output)
    binary_mask[output > 0] = 1

    # Convert the binary mask to a PIL image and return it
    binary_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
    return binary_mask

import matplotlib.colors as colors

def plot_segmentation(preds):
    colors = np.array([
    [0, 0, 0],      # 0: Background (Black)
    [128, 0, 0],    # 1: Aeroplane (Maroon)
    [0, 128, 0],    # 2: Bicycle (Green)
    [128, 128, 0],  # 3: Bird (Olive)
    [0, 0, 128],    # 4: Boat (Navy)
    [128, 0, 128],  # 5: Bottle (Purple)
    [0, 128, 128],  # 6: Bus (Teal)
    [128, 128, 128],# 7: Car (Gray)
    [64, 0, 0],     # 8: Cat (Brown)
    [192, 0, 0],    # 9: Chair (Red)
    [64, 128, 0],   # 10: Cow (Olive Green)
    [192, 128, 0],  # 11: Dining table (Orange)
    [64, 0, 128],   # 12: Dog (Purple)
    [192, 0, 128],  # 13: Horse (Magenta)
    [64, 128, 128], # 14: Motorbike (Cyan)
    [192, 128, 128],# 15: Person (Pink)
    [0, 64, 0],     # 16: Potted plant (Dark Green)
    [128, 64, 0],   # 17: Sheep (Dark Orange)
    [0, 192, 0],    # 18: Sofa (Lime)
    [128, 192, 0],  # 19: Train (Light Orange)
    [0, 64, 128],   # 20: TV/monitor (Dark Blue)
])

    # Normalize the predictions to [0, 1]
    preds = preds / 21.0

    # Create a color image from the predictions tensor
    color_image = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(21):
        color_image[preds[i] > 0.5] = colors[i]

    # Show the color image
    plt.imshow(color_image)
    plt.axis('off')
    plt.show()
    
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
import torchvision.transforms.functional as F

def detection_mask_rcnn(image):
    origin_img = image.copy()
    img_arr = np.array(image) 
    to_tens_t = transforms.ToTensor()
    image = to_tens_t(img_arr)
    print(image.shape)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    _transforms = weights.transforms()
    t_img = [_transforms(image)]

    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    with torch.no_grad():
        output = model([image])

    score_threshold = 0.1
    boxes = output[0]['boxes'][output[0]['scores'] > score_threshold]
    labels = output[0]['labels'][output[0]['scores'] > score_threshold]
    scores = output[0]['scores'][output[0]['scores'] > score_threshold]
    _pil_to_tensor = transforms.PILToTensor()
    img_w_bounding_boxes = draw_bounding_boxes(_pil_to_tensor(origin_img).to(dtype=torch.uint8),
                                               boxes,
                                               width=4)
    show([img_w_bounding_boxes])
    
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def segmentation_mask_rcnn(image):
    origin_img = image.copy()
    img_arr = np.array(image) 
    to_tens_t = transforms.ToTensor()
    image = to_tens_t(img_arr)
    print(image.shape)
    
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights, progress=False)
    model = model.eval()
    
    batch = torch.stack([image])
    with torch.no_grad():
        output = model(batch)['out']
        
    
