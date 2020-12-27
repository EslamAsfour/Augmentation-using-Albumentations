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


'''
 Function takes imgs Directory and Json file and return 
        Dict {
                "img_Name" : TargetImg ,
                "img_RGB" : Img_In_RGB,
                "bbox": [],
                "class_labels": []
                }
'''     
def Get_Prep_Annotation(imgDir,JsonPath):
    #Get list of Imgs
    
    images_file = os.listdir(imgDir)
    #Get JSON File

    f = open(JsonPath)
    Annotation_Json = json.load(f) 
    Annotation_Output= []

    #Get img ID
    for TargetImg in images_file:
        
        #Read img
        img_in_RGB =cv2.imread(imgDir+f"/{TargetImg}")
        img_in_RGB = cv2.cvtColor(img_in_RGB,cv2.COLOR_BGR2RGB)

        Target_ID = None
        #Search JSON file for img Id 
        for img in Annotation_Json['images']:
            if img['file_name'] == TargetImg:
                Target_ID = img['id']
                break
        # Search with Img_id to get every bbox of the img
     
        Img_Dic = {
            "img_Name" : TargetImg ,
            "img_RGB" : img_in_RGB,
            "bbox": [],
            "class_labels": []
        }
        
        # Creat dic for every img
        for Annotation in Annotation_Json['annotations']:
            if Annotation['image_id'] == Target_ID:
                Img_Dic["bbox"].append(Annotation['bbox'])
                Img_Dic["class_labels"].append(Annotation['category_id'])
                   
        Annotation_Output.append(Img_Dic)
        
    return Annotation_Output

        
            
                
                

            
            
                
            

        
        



