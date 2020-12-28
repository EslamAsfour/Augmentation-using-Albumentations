import cv2
from matplotlib import pyplot as plt
import albumentations as A

import os 
import json



BOX_COLOR = (255, 0, 0)      # Red
TEXT_COLOR = (255, 255, 255) # White

"""
    Visualizes a single bounding box on the image
"""
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    # to add every bbox to the img
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    
    # imshow works with BGR not RGB
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow("Out",img)
    cv2.waitKey(0)
    
'''
    Function to rename all imgs to [1.jpg , 2.jpg, ...]
        Function takes 
            1- images Directory path
            2- Annotations Json File
'''  
    
def Rename_COCO(imgs_Path , Annotation_Path):
    
    #read Annotations Json file
    f = open(Annotation_Path)
    Annotations = json.load(f) 
    
    #change directory to the imgs dir
    os.chdir(imgs_Path)
    
    count = 0
    #Loop over every img_name in the annotations and 
    for img in Annotations['images']:
        original_Name = img['file_name']
        #Change name in annotations
        img['file_name'] = str(count)+".jpg"
        # Change name in directory
        os.rename(original_Name,str(count)+".jpg")
        
        #inc the counter
        count+= 1
    # Save Annotations changes
    with open(Annotation_Path, 'w') as outfile:
        json.dump(Annotations, outfile)