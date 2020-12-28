import os
import re


def parse_yolo_file():
    class_labels = []
    bboxes = []
    with open('Annotations_Exp/1_aug.txt') as f:
        content = f.readlines()
    for line in content:
        line.strip()
        strings = line.split()
        # print(strings)
        class_labels.append(int(strings[0]))
        strings.pop(0)
        strings = list(map(float, strings))
        print(strings)
        bboxes.append(strings)

    return class_labels, bboxes
    # print(content)


def parse_yolo_files():
    # os.chdir('C:/Users/Omar Magdy/PycharmProjects/Augmentation_Exp/Annotations_Exp/')
    os.chdir('D:/My Folders/University/GP/Yolov5/train/labels')
    files = os.listdir()
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    files.sort(key=alphanum_key)

    s_class_labels = []
    s_bboxes = []
    m_class_labels = []
    m_bboxes = []
    for file in files:
        with open(file) as f:
            content = f.readlines()
        for line in content:
            line.strip()
            strings = line.split()
            # print(strings)
            s_class_labels.append(int(strings[0]))
            strings.pop(0)
            strings = list(map(float, strings))
            # print(strings)
            s_bboxes.append(strings)
        m_class_labels.append(s_class_labels)
        m_bboxes.append(s_bboxes)
        s_class_labels = []
        s_bboxes = []
    return m_class_labels, m_bboxes
    # print(content)
