# Augmentation-using-Albumentations for COCO and YOLO format Datasets

## 1) COCO Format 

## - Content 
- Review COCO Format 
- Prepare DS to use Albumentation
- Prepare Transformations
- Save output Annotations and imgs

## 1) Review COCO Format
```json
  {
    "info": {
        "year": "2020",
        "version": "",
        "description": "",
        "contributor": "",
        "url": "",
        "date_created": ""
    },
    "licenses": [
        {
            "id": 1,
            "url": "",
            "name": "Unknown"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "",
            "supercategory": ""
        },
        ...
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0.jpg",
            "height": 416,
            "width": 416,
            "date_captured": ""
        },
        ...
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": ,
            "bbox": [
                 ,
                 ,
                 ,
                
            ],
            "area": ,
            "segmentation": [],
            "iscrowd": 0
        }
        ...
    ]
  { 
```
<br>

## 2) Prepare DS to use Albumentation

  - To perfome any Transformations with Albumentation you need to input the transformation function inputs as shown :
   1- Image in RGB  = (list)[]
   2- Bounding boxs : (list)[]
   3- Class labels : (list)[]
   4- List of all the classes names for each label 
  
  ### Example :
     - img_RGB : array([[[ 71, 121, 130].....)
     - bbox : [175, 194, 21.5, 30.5]
     - class_label : [3] -> 'buoy_red'
     - Classes_Name = ['Objects','buoy_green','buoy_orange','buoy_red','buoy_yellow','gate_edge']
      <br>
  ## Our Functions 
  #### Function ``` Get_Prep_Annotation(imgDir,JsonPath) ``` return the needed input form as a dic 
  #### Example : 
  ```python 
        Dict {
                "img_Name" : TargetImg ,
                "img_RGB" : Img_In_RGB,
                "bbox": [],
                "class_labels": []
                }
  ```


<br>



