import os
import sys
import shutil

def gendir():
    base = '/train_tiny_data/train_data/food-101/'
    labels = base + 'meta/classes.txt'
    if not os.path.exists(base + 'val'):
        os.mkdir(base + 'val')
    with open(labels) as f:
        for line in f.readlines():
            dir_name = base + 'val/' + line.strip('\n')
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

    # mv images to val
    tests = base + 'meta/test.txt'
    with open(tests) as f:
        for line in f.readlines():
            file_name = line.strip('\n') + '.jpg'
            file_src = base + 'images/' + file_name
            mv_file_dst = base + 'val/' + file_name
            if os.path.exists(file_src):
                shutil.move(file_src, mv_file_dst)
    # rename images
    os.rename(base+'images', base+'train')

    print('process finish!')


if __name__ == '__main__':
    gendir()




