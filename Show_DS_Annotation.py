from DataSet_Prep.Prepare_COCO import Get_Prep_Annotation
from DataSet_Prep.util import *
import cv2


## Put ur Paths
imgDir_Path = "E:/AUV/DataSet Creation/Final DS/train"
Annotation_File = "E:/AUV/DataSet Creation/Final DS/Train_Annotation.json"
#########################
# Get Dict with {
#       img_RGB , bbox , class_labels  
#}

Imgs_Prepared_to_Trans = Get_Prep_Annotation(imgDir_Path,Annotation_File)
#########################
# Put ur classes names 
Label_2_ClassName =['Objects','buoy_green','buoy_orange','buoy_red','buoy_yellow','gate_edge']

for img in Imgs_Prepared_to_Trans:
    print(img['img_Name'])
    visualize(img['img_RGB'],img['bbox'],img['class_labels'],Label_2_ClassName)
