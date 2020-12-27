from DataSet_Prep.Prepare_COCO import Get_Prep_Annotation
from DataSet_Prep.util import *
import cv2


## Put ur Paths
imgDir_Path = "E:\AUV\DataSet Creation\Augmentation\img"
Annotation_File = "E:\AUV\DataSet Creation\Augmentation\output_2.json"
#########################
# Get Dict with {
#       img_RGB , bbox , class_labels  
#}

Imgs_Prepared_to_Trans = Get_Prep_Annotation(imgDir_Path,Annotation_File)
#########################
# Put ur classes names 
Label_2_ClassName =['','gates','bin','bin_cover','buoy_green','buoy_orange','buoy_red','buoy_yellow','channel','object_dropoff','object_pickup','qual_gate','torpedo_cover','torpedo_hole','torpedo_target']

for img in Imgs_Prepared_to_Trans:
    print(img['img_Name'])
    visualize(img['img_RGB'],img['bbox'],img['class_labels'],Label_2_ClassName)
