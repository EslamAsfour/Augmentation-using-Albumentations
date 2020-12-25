import albumentations as a
import cv2
import os
from math import floor
import re


def sort_images():
    os.chdir('D:/My Folders/University/GP/Gate_Task/Gate_Task_Training')
    image_files = os.listdir()

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    image_files.sort(key=alphanum_key)
    print(image_files)


def transform_multiple_images_multiple_transforms(tfs):
    os.chdir('D:/My Folders/University/GP/Gate_Task/Gate_Task_Training')
    images_files = os.listdir()

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    images_files.sort(key=alphanum_key)

    images = []
    for image_file in images_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    transformed_images = []
    for tf in tfs:
        for image in images:
            transformed = tf(image=image)
            transformed_image = transformed["image"]
            transformed_images.append(transformed_image)
    images_size = len(images)
    for index, transformed_image in enumerate(transformed_images):
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{(index % images_size) + 1}_aug{floor(index/images_size) + 1}.jpg", transformed_image)


def transform_multiple_images(tf):
    images_files = os.listdir('Images/')
    images = []
    for image_file in images_files:
        image = cv2.imread(f'Images/{image_file}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    transformed_images = []
    for image in images:
        transformed = tf(image=image)
        transformed_image = transformed["image"]
        transformed_images.append(transformed_image)
    os.chdir('Images/')
    for index, transformed_image in enumerate(transformed_images):
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{index + 1}_aug.jpg", transformed_image)


def transform_single_image(tf):
    # Read an image with OpenCV and convert it to the RGB color space
    image = cv2.imread("Images/1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = tf(image=image)
    transformed_image = transformed["image"]

    os.chdir("Images/")
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("1_aug.jpg", transformed_image)
