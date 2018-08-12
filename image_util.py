# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image


def image_array(image_path):
    image = Image.open(image_path)
    return np.array(image)


def image_split(image_path, steps=500):
    img_array = image_array(image_path)
    image_shape = img_array.shape
    batch = round(image_shape[1] / steps)
    for i in range(batch):
        im_array = img_array[:, i * steps:(i + 1) * steps, :]
        img = Image.fromarray(im_array.astype('uint8')).convert('RGB')
        # if image_shape[0]
        img = img.transpose(Image.ROTATE_270)
        img.save(f'split_image/{i}.jpg')


def image_show():
    image = Image.open('1.jpg')
    image = image.transpose(Image.ROTATE_270)
    image_array = np.array(image)
    print(image_array.shape)
    image.show()


if __name__ == '__main__':
    image_split('1.jpg')
    image_show()
