# Augmentation-using-Albumentations for COCO and YOLO format Datasets

## 1) COCO Format 

## - Content 
- Review COCO Format 
- Prepare DS to use Albumentation
- Choose your Transformations
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
   1- Image in RGB  = (list)[ ]
   2- Bounding boxs : (list)[ ]
   3- Class labels : (list)[ ]
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
                "class_labels": [ ]
                }
  ```


<br>

## 3) Choose your Transformations

### In Test.py you will find a sample of our transformations 

```python
transform1 = a.Compose([
    a.RandomCrop(width=256, height=256),
    a.HorizontalFlip(p=1),
    a.RandomBrightnessContrast(p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform2 = a.Compose([
    a.Blur(blur_limit=2, p=1),
    a.ChannelShuffle(p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))
...

transforms = [transform1,transform2, ... ]

```
#### Then we call the Function :
```python 
  Aug_IMGs( In_img_Path , In_Annotation_Path , Classes_Name , Transforms , Out_Img_Path , Out_Annotation_Path )
```


## 4) Save output Annotations and imgs:

### What should we save ?
   ### - Augmented image in RGB
```python
      cv2.imwrite(out_Img_P + new_Name , cv2.cvtColor(output_Aug['image'],cv2.COLOR_RGB2BGR) )
```
  ###  - Add in "images" Entry in Json the new image
```python
    Out_Json['images'].append({
                "file_name": new_Name,
                "height": output_Aug['image'].shape[0],
                "width": output_Aug['image'].shape[1],
                "id": Img_Id_Start
            })
```
   ### - Add in "annotations" Entry in Json the new Bbox
```python
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
```



