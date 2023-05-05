from DataSet_Prep.Prepare_COCO import Get_Prep_Annotation,Aug_IMGs
import albumentations as a
import sys
import cv2
from DataSet_Prep.util import *
import json

##################################################################
##################### Transformations ############################
transform1 = a.Compose([
    a.RandomCrop(width=256, height=256),
    a.HorizontalFlip(p=1),
    a.RandomBrightnessContrast(p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform2 = a.Compose([
    a.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
    a.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform3 = a.Compose([
    a.Blur(blur_limit=2, p=1),
    a.ChannelShuffle(p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform4 = a.Compose([
    a.GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=1),
    a.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.5, p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform5 = a.Compose([
    a.MedianBlur(blur_limit=3, p=1),
    a.FancyPCA(alpha=0.1, p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform6 = a.Compose([
    a.MotionBlur(blur_limit=3, p=1),
    a.ToSepia(p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform7 = a.Compose([
    a.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast', p=1),
    a.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform8 = a.Compose([
    a.Resize(width=320, height=320),
    a.VerticalFlip(p=1),
    a.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform9 = a.Compose([
    a.Resize(width=352, height=352, interpolation=cv2.INTER_LINEAR, p=1),
    a.RandomRotate90(p=1),
    a.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform10 = a.Compose([
    a.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), p=1),
    a.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
    a.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transforms = [transform1, transform2, transform3, transform4, transform5, transform6, transform7, transform8,
              transform9, transform10]


##################################################################



in_Path ="img/"
in_Ann_Path ="output_2.json" 
Out_Path = "E:/AUV/DataSet Creation/Augmentation/Out_img/"
out_Annotation_p = "output_2.json" 



Classes_Name = ["Gate", "Any"]

#Rename_COCO(in_Path,in_Ann_Path)

Aug_IMGs(in_Path, in_Ann_Path, Classes_Name, transforms, Out_Path, out_Annotation_p)
    
        
        
        
    
    
    
    
    
    
    
    



    
