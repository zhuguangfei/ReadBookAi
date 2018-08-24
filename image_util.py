# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import os


def image_array(image_path):
    image = Image.open(image_path)
    return np.array(image)


def image_split(image_path, image_name, steps=500):
    img_array = image_array(image_path)
    img_array = img_array[7:3010, 15:4019, :]
    index = 0
    for i in range(3):
        im_height = img_array[i * 1000 : (i + 1) * 1000, :, :]
        for j in range(8):
            im_widht = im_height[:, j * 500 : (j + 1) * 500, :]
            img = Image.fromarray(im_widht.astype('uint8')).convert('RGB')
            img = img.transpose(Image.ROTATE_270)
            img.save(f'split_image/{image_name}_{index}.jpg')
            f = open(f'image_text/{image_name}_{index}.txt', 'w')
            f.close()
            index += 1


def image_show():
    image = Image.open('1.jpg')
    image = image.transpose(Image.ROTATE_270)
    image_array = np.array(image)
    print(image_array.shape)
    image.show()


def image_split_two(image_path, image_name):
    img_array = image_array(image_path)
    for j in range(2):
        if j == 0:
            im_widht = img_array[:, 0:500, :]
        else:
            im_widht = img_array[:, 500:, :]

        img = Image.fromarray(im_widht.astype('uint8')).convert('RGB')
        img.save(f'split_image_two/{image_name}_{j}.jpg')


if __name__ == '__main__':
    listdir = os.listdir('split_image')
    for l in listdir:
        image_path = os.path.join('split_image', l)
        image_split_two(image_path, l.split('.')[0])
