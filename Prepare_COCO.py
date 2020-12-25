import albumentations as a
import cv2
import os
import json

'''

    Python Script to prepare the dataset in COCO format to be Augmented using Albumentations libirary
    
'''
'''
    COCO Format :
 {
    "images": [
        {
            "file_name": "",
            "height": ,
            "width": ,
            "id": 
        },
        ...
    "annotations": [
        {
            "area": ,
            "iscrowd": ,
            "image_id": ,
            "bbox": [
                215,
                185,
                294,
                156
            ],
            "category_id": 0,
            "id": 1,
            "ignore": 0,
            "segmentation": []
        }
        ...
    "categories": [
        {
            "supercategory": "",
            "id": ,
            "name": ""
        }
    }
'''


#Get list of Imgs
images_file = os.listdir('img/')
#Get JSON File
Annotation_Json = json.load(f) 
#Get img ID
Output

for TargetImg in images_file:
    Target_ID = None
    #Search JSON file for img Id 
    for img in Annotation_Json['images']:
        if img['file_name'] == TargetImg:
            Target_ID = img['id']
            break
    # Search with Img_id to get every bbox of the img

    Img_Dic = {
        "bbox": [],
        "class_labels ": []
    }
    
    # Creat dic for every img
    for Annotation in Annotation_Json['annotations']:
        if Annotation['image_id'] == Target_ID:
            Img_Dic['bbox'].append(Annotation['bbox'])
            Img_Dic['class_labels'].append(Annotation['category_id'])
            
            

        
        
            
         

    
    



