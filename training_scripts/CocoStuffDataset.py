from torch.utils.data import Dataset
import itertools
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os
import scipy.io

class CocoStuffDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instace_segmap_root,
        instance_prompt,
        coco_stuff_labels_file,
        tokenizer=None,
        image_list_path=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize
        self.color_jitter = color_jitter
        self.prompt = instance_prompt
        
        self.classes_dict = self.__get_cocostuff_classes_dict__(coco_stuff_labels_file)
        
        if image_list_path:
            # read image list from file into a list
            with open(image_list_path, 'r') as f:
                image_list = [line.strip() for line in f]
            self.instance_imgs_paths = [os.path.join(instance_data_root, img +'.jpg') for img in image_list]
            self.instance_segmap_paths = [os.path.join(instace_segmap_root, img +'.mat') for img in image_list]
        else:
            # iterate all files in instance_data_root and add their path to the list if it has .jpg extension
            self.instance_imgs_paths = [os.path.join(instance_data_root, filename) \
                for filename in os.listdir(instance_data_root) if filename.endswith(".jpg")]
            self.instance_segmap_paths = [os.path.join(instace_segmap_root, os.path.splitext(os.path.basename(path))[0] + '.mat') \
                for path in self.instance_imgs_paths]
        
        self.classes = self.__get_classes__()
        self.instance_segmap_paths = self.__duplicate_paths_according_to_classes__(self.instance_segmap_paths, self.classes)
        self.instance_imgs_paths = self.__duplicate_paths_according_to_classes__(self.instance_imgs_paths, self.classes)
        
        merged = list(itertools.chain(*self.classes))
        self.classes = merged
        
        assert len(self.classes) == len(self.instance_imgs_paths) == len(self.instance_segmap_paths), \
            "number of class annotations is different from number of segmaps or images!"
        self.__data_len__ = len(self.classes)
        
        img_transforms = []
        segmap_transforms = []

        if resize:
            resize_t = transforms.Resize(
                    (size,size), interpolation=transforms.InterpolationMode.BILINEAR
                )
            img_transforms.append(resize_t)
            segmap_transforms.append(transforms.Resize(
                    (size, size), interpolation=transforms.InterpolationMode.NEAREST
                )
            )
        
        if center_crop:
            center_crop_t = transforms.CenterCrop(size)
            img_transforms.append(center_crop_t)
            segmap_transforms.append(center_crop_t)

        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        
        if h_flip:
            h_flip_t = transforms.RandomHorizontalFlip()
            img_transforms.append(h_flip_t)
            segmap_transforms.append(h_flip_t)

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
            ]
        )
        self.segmap_transforms = transforms.Compose(
            [*segmap_transforms]
        )
    
    def __len__(self):
        return self.__data_len__
                    
    def __get_cocostuff_classes_dict__(self, filename):
        class_dict = {}

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()  # remove any whitespace or newlines at the beginning or end
                if line:  # make sure the line is not empty
                    number, class_name = line.split(": ")
                    class_dict[class_name] = int(number)
                    class_dict[int(number)] = class_name

        return class_dict

    '''
    classes returned is the number corelating to the class in the coco stuff dataset
    '''
    def __get_classes__(self):
        classes = []
        for segmap_path in self.instance_segmap_paths:
            seg_file = scipy.io.loadmat(segmap_path)
            seg_arr = seg_file['S']
            _img_classes = np.unique(seg_arr)
            # remove 0's from img_classes
            img_classes = [x for x in _img_classes if x != 0]
            classes.append(img_classes)
        
        return classes
            
    def __duplicate_paths_according_to_classes__(self, img_list, classes):
        new_img_list = []
        for i, img_path in enumerate(img_list):
            for class_id in classes[i]:
                new_img_list.append(img_path)
        
        return new_img_list
    
    '''
        Gets png image of shape (h,w) and coco_class
        Returns a torch tensor where all pixel that arent the class are (0,0,0) and (255,255,255) for the class
    '''
    def __get_bin_segmap_by_class__(self, img, coco_class):
        seg_img = np.asarray(img).copy()
        new_seg_img = seg_img
        
        if isinstance(coco_class, str):
            class_pixel = self.classes_dict[coco_class]
        else:
            class_pixel = coco_class

        new_seg_img[new_seg_img != class_pixel] = 0
        new_seg_img[new_seg_img != 0] = 255

        new_seg_img = np.repeat(new_seg_img[:, :, np.newaxis], 3, axis=2)
        new_seg_img = torch.from_numpy(new_seg_img).permute(2, 0, 1)

        return new_seg_img
    
    def __getitem__(self, index):
        example = {}
        img_path = self.instance_imgs_paths[index % self.__data_len__]
        instance_image = Image.open(
            img_path
        )
        example['instance_images_path'] = str(img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        ## Segmap
        segmap_img_path = self.instance_segmap_paths[index % self.__data_len__]
        segmap_instance_image = Image.fromarray(scipy.io.loadmat(segmap_img_path)['S'])
        segmap_instance_image = self.segmap_transforms(segmap_instance_image)

        _class = self.classes[index % self.__data_len__]
        # segmap_instance_image.save(f'{voc_class}_before.jpeg')
        
        segmap_instance_tensor = self.__get_bin_segmap_by_class__(segmap_instance_image, _class)
        # segmap_instance_image.save(f'{voc_class}.jpeg')

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_segmap_images"] = segmap_instance_tensor
        example["instance_classes"] = self.classes_dict[_class]
        if self.tokenizer:
            example["instance_prompt_ids"] = self.tokenizer(
                self.prompt + " " + self.classes_dict[_class],
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        
        return example