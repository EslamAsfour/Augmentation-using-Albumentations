import albumentations as a
import cv2
import os
import json
from DataSet_Prep.util import *


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

        
            
def Aug_IMGs(in_img_P,in_Annotation_P ,in_Label_2_ClassName,tfs,out_Img_P,out_Annotation_p):
    # Get Dict with {
    #       img_RGB , bbox , class_labels  
    #}
    Imgs_Prepared_to_Trans = Get_Prep_Annotation(in_img_P,in_Annotation_P)
    
    f = open(in_Annotation_P)
    Out_Json = json.load(f) 
    # Get last Img_id to start counting from
    Img_Id_Start = Out_Json['images'][-1]['id']
    #Get Last Annotation id
    Annotation_Id_Start =Out_Json['annotations'][-1]['id']

    Label_2_ClassName = in_Label_2_ClassName
    tf_Count = 1
    # loop over all the imgs Dict
    for img in Imgs_Prepared_to_Trans:
        # Every img go through every transform
        
        tf_Count = 1
        for tf in tfs:
            
            #Augmentation
            output_Aug = (tf(image=img['img_RGB'], bboxes=img['bbox'], category_ids=img['class_labels']))
            
            ## Save Output img in RGB 
            new_Name = f"Aug_{tf_Count}_{img['img_Name']}"
            tf_Count += 1
            cv2.imwrite(out_Img_P + new_Name , cv2.cvtColor(output_Aug['image'],cv2.COLOR_RGB2BGR) )
            Img_Id_Start+= 1
            ## Add Annotation to JSON
            ## Add img to images
            Out_Json['images'].append({
                "file_name": new_Name,
                "height": output_Aug['image'].shape[0],
                "width": output_Aug['image'].shape[1],
                "id": Img_Id_Start
            })
            
            ## Add Annotation to annotations
            ## Note that we can have multiple Annotations in one img so we need to loop through all f the bboxs and append each one
            for bbox , cat_it in zip(output_Aug['bboxes'],output_Aug['category_ids']):
                # Convert the numbers to ints
                bbox = list(map(int, bbox))
                Annotation_Id_Start += 1
                Out_Json['annotations'].append({
                    "area": int(bbox[-1]*bbox[-2]),
                    "iscrowd": 0,
                    "image_id": Img_Id_Start,
                    "bbox": bbox,
                    "category_id": int(cat_it),
                    "id": Annotation_Id_Start,
                    "ignore": 0,
                    "segmentation": []
                })
        
        
        
            ##Show Output
            #visualize(output_Aug['image'],output_Aug['bboxes'],output_Aug['category_ids'],Label_2_ClassName)
    
    with open(out_Annotation_p, 'w') as outfile:
        json.dump(Out_Json, outfile)
        
               
                

            
            
                
            

        
        



