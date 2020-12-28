import os


def rename_files():
    os.chdir('D:/My Folders/University/GP/Yolov5/train/images')
    files = os.listdir()
    for file in files:
        new_file = file.split('_')[0]
        os.rename(file, f'{new_file}.jpg')


rename_files()
