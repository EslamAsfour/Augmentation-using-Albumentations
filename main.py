from my_yolo_parse import *
from my_transforms import *
import albumentations as a

m_class_labels, m_bboxes = parse_yolo_files()

transform1 = a.Compose([
    a.RandomCrop(width=256, height=256),
    a.HorizontalFlip(p=1),
    a.RandomBrightnessContrast(p=1)
], bbox_params=a.BboxParams(format='yolo', label_fields=['class_labels']))

transform2 = a.Compose([
    a.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
    a.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)
])

transform3 = a.Compose([
    a.Blur(blur_limit=2, p=1),
    a.ChannelShuffle(p=1)
])

transform4 = a.Compose([
    a.GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=1),
    a.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.5, p=1)
])

transform5 = a.Compose([
    a.MedianBlur(blur_limit=3, p=1),
    a.FancyPCA(alpha=0.1, p=1)
])

transform6 = a.Compose([
    a.MotionBlur(blur_limit=3, p=1),
    a.ToSepia(p=1)
])

transform7 = a.Compose([
    a.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast', p=1),
    a.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1)
])

transform8 = a.Compose([
    a.Resize(width=320, height=320),
    a.VerticalFlip(p=1),
    a.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
])

transform9 = a.Compose([
    a.Resize(width=352, height=352, interpolation=cv2.INTER_LINEAR, p=1),
    a.RandomRotate90(p=1),
    a.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1)
])

transform10 = a.Compose([
    a.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), p=1),
    a.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
    a.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
])

transforms = [transform1, transform2, transform3, transform4, transform5, transform6, transform7, transform8,
              transform9, transform10]


# transform_single_image_with_bboxes(transform1, m_class_labels[0], m_bboxes[0])
# transform_multiple_images_with_bboxes(transform1, m_class_labels, m_bboxes)


# Trials #

# for element in dir(a):
#     print(help(element))

# print(dir(a))

# parse_yolo_file()

# os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Images_Exp/')
# image = cv2.imread('Images_Exp/1_aug.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# category_ids, bboxes = parse_yolo_file()
# category_id_to_name = {8: 'qual_gate', 3: 'buoy_red', 4: 'buoy_yellow', 2: 'buoy_green'}
#
# visualize(image, bboxes, category_ids, category_id_to_name)

# print(m_class_labels)
# print(m_bboxes)
# transform_single_image(transform1)
# transform_multiple_images(transform1)
# transform_multiple_images_multiple_transforms(transforms)
# sort_images()
