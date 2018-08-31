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


def image_split_util(image_path, image_name, output_path, width=2, height=2):
    img_array = image_array(image_path)
    shape = img_array.shape
    width_step = int(shape[1] / width)
    height_step = int(shape[0] / height)
    # print(shape, width_step, height_step)
    index = 0
    for i in range(width):
        width_array = img_array[:, i * width_step : (i + 1) * width_step, :]
        for j in range(height):
            height_array = width_array[j * height_step : (j + 1) * height_step, :, :]
            img = Image.fromarray(height_array.astype('uint8')).convert('RGB')
            img.save(f'{output_path}/{image_name}_{index}.jpg')
            index += 1


if __name__ == '__main__':
    parent_path = 'source'
    output_parent_path = 'source_two'
    listdir = os.listdir(parent_path)
    for l in listdir:
        images = os.listdir(os.path.join(parent_path, l))
        output_path = os.path.join(output_parent_path, l)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in images:
            image_path = os.path.join(parent_path, l, image)
            image_split_util(image_path, image.split('.')[0], output_path)
