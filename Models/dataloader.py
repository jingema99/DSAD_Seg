import os, sys
from torch._C import dtype
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as vutils
import torch
import numpy as np
import math
import cv2
import albumentations
import multiprocessing

class CobotLoaderBinary(Dataset):

    def __init__(self, root_dir, label, num_labels, transform, image_size=None, id=-1, create_negative_labels=False):
        self.root_dir = root_dir
        self.images = []
        self.labels = []

        self.label = label

        self.transform = transform
        self.create_negative_labels = create_negative_labels

        self.num_pixels = 0
        self.num_bg_pixels = 0
        self.files = []
        self.a_masks = []

        self.id = id
        self.num_labels = num_labels
    
        for file in os.listdir(root_dir):
            if "png" in file and file[0:5] == "image": 
                file = os.path.join(root_dir, file)
                img = cv2.imread(file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.files.append(file)
   
                if image_size is not None:
                    img = cv2.resize(img,image_size)

                mask_file = file.replace("image", "mask")
                mask_orig = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
                
                if image_size is not None:
                    mask_orig = cv2.resize(mask_orig,image_size, interpolation=cv2.INTER_NEAREST)
                
                self.num_bg_pixels += np.sum(mask_orig == 0)
                self.num_pixels += mask_orig.shape[0]*mask_orig.shape[1]

                mask = (mask_orig > 0)*1

                self.images.append(img)
                self.labels.append(mask)

    def get_frequency(self):
        return self.num_bg_pixels, self.num_pixels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]

        if self.create_negative_labels:
            masks = []
            mask_dict = {}
            mask = self.labels[idx]
            mask_dict["mask"] = mask

            neg_mask = mask - 1   

            for i in range(self.num_labels):
                str = "mask%d" % i
                if i == self.label - 1:
                    c_mask = mask
                else:
                    c_mask = neg_mask
                mask_dict[str] = c_mask

            transformed = self.transform(image=self.images[idx], **mask_dict)

            for i in range(self.num_labels):
                str = "mask%d" % i
                masks.append(transformed[str])
            
            mask = np.stack(masks, axis=0)
            img = transformed["image"]
            mask_orig = transformed["mask"]

            return img, mask, self.label, mask_orig
            
        if not self.transform is None:
            transformed = self.transform(image=img, mask=mask)

            img = transformed["image"]
            mask = transformed["mask"]
      
        return img, mask, self.label
        