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


def transform_single_image_with_bboxes(tf, class_labels, bboxes):
    # Read an image with OpenCV and convert it to the RGB color space
    os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Images_Exp/')
    image = cv2.imread("1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = tf(image=image, class_labels=class_labels, bboxes=bboxes)
    transformed_image = transformed["image"]
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    # os.chdir("Images_Exp/")
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("1_aug.jpg", transformed_image)

    os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Annotations_Exp/')
    with open('1_aug.txt', 'a') as f:
        for i in range(len(transformed_class_labels)):
            f.write(f'{transformed_class_labels[i]} ')
            transformed_bbox = transformed_bboxes[i]
            for j in range(len(transformed_bbox) - 1):
                f.write(f'{transformed_bbox[j]} ')
            if i != len(transformed_class_labels) - 1:
                f.write(f'{transformed_bbox[-1]}\n')
            else:
                f.write(f'{transformed_bbox[-1]}')


def transform_multiple_images_with_bboxes(tf, m_class_labels, m_bboxes):
    os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Images_Exp/')
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
    transformed_m_bboxes = []
    transformed_m_class_labels = []
    for index, image in enumerate(images):
        transformed = tf(image=image, class_labels=m_class_labels[index], bboxes=m_bboxes[index])
        transformed_image = transformed["image"]
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']
        transformed_images.append(transformed_image)
        transformed_m_bboxes.append(transformed_bboxes)
        transformed_m_class_labels.append(transformed_class_labels)

    for index, transformed_image in enumerate(transformed_images):
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{images_files[index].split('.')[0]}_aug.jpg", transformed_image)

    print(transformed_m_class_labels)
    print(transformed_m_bboxes)

    os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Annotations_Exp/')
    for index, image_file in enumerate(images_files):
        with open(f"{image_file.split('.')[0]}_aug.txt", 'a') as f:
            for i in range(len(transformed_m_class_labels[index])):
                f.write(f'{transformed_m_class_labels[index][i]} ')
                transformed_bbox = transformed_m_bboxes[index][i]
                for j in range(len(transformed_bbox) - 1):
                    f.write(f'{transformed_bbox[j]} ')
                if i != len(transformed_m_class_labels[index]) - 1:
                    f.write(f'{transformed_bbox[-1]}\n')
                else:
                    f.write(f'{transformed_bbox[-1]}')


def transform_multiple_images_multiple_transforms_with_bboxes(tfs, m_class_labels, m_bboxes):
    os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Images_Exp/')
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
    transformed_m_bboxes = []
    transformed_m_class_labels = []
    for tf in tfs:
        for index, image in enumerate(images):
            transformed = tf(image=image, class_labels=m_class_labels[index], bboxes=m_bboxes[index])
            transformed_image = transformed["image"]
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            transformed_images.append(transformed_image)
            transformed_m_bboxes.append(transformed_bboxes)
            transformed_m_class_labels.append(transformed_class_labels)

    images_size = len(images)
    for index, transformed_image in enumerate(transformed_images):
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{images_files[index % images_size].split('.')[0]}_aug{floor(index/images_size) + 1}.jpg", transformed_image)

    print(transformed_m_class_labels)
    print(transformed_m_bboxes)

    os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Annotations_Exp/')
    for index in range(len(transformed_images)):
        with open(f"{images_files[index % images_size].split('.')[0]}_aug{floor(index/images_size) + 1}.txt", 'a') as f:
            for i in range(len(transformed_m_class_labels[index])):
                f.write(f'{transformed_m_class_labels[index][i]} ')
                transformed_bbox = transformed_m_bboxes[index][i]
                for j in range(len(transformed_bbox) - 1):
                    f.write(f'{transformed_bbox[j]} ')
                if i != len(transformed_m_class_labels[index]) - 1:
                    f.write(f'{transformed_bbox[-1]}\n')
                else:
                    f.write(f'{transformed_bbox[-1]}')

