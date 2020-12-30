# Augmentation-using-Albumentations for COCO and YOLO format Datasets

## 1) COCO Format 

## - Content 
- Review COCO Format 
- Prepare DS to use Albumentation
- Choose your Transformations
- Save output Annotations and imgs
- Sample Output with Transformations

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

## 5) Sample Output with Transformations
```python 
transform1 = a.Compose([
    a.RandomCrop(width=256, height=256),
    a.HorizontalFlip(p=1),
    a.RandomBrightnessContrast(p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))

transform2 = a.Compose([
    a.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
    a.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)
], bbox_params=a.BboxParams(format='coco', label_fields=['category_ids']))


```


Original                        |Aug_1                      |Aug_2              
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/152.jpg)  |  ![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_1_152.jpg) |![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_2_152.jpg) |

```python
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

```


Aug_3                        |Aug_4                     | Aug_5            
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_3_152.jpg)  |  ![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_4_152.jpg) |![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_5_152.jpg)

```python 
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

```


Aug_6                        |Aug_7                     | Aug_8            
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_6_152.jpg)  |  ![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_7_152.jpg) |![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_8_152.jpg)

```python 
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

```

Aug_9                        |Aug_10                          
:-------------------------:|:-------------------------:|
![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_9_152.jpg)  |  ![](https://github.com/EslamAsfour/Augmentation-using-Albumentations/blob/main/sample_Output/Aug_10_152.jpg) |



