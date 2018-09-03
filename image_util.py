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
        im_height = img_array[i * 1000: (i + 1) * 1000, :, :]
        for j in range(8):
            im_widht = im_height[:, j * 500: (j + 1) * 500, :]
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


def image_split_util(image_path, image_name, output_path, width=3, height=3):
    img_array = image_array(image_path)
    shape = img_array.shape
    width_step = int(shape[1] / width)
    height_step = int(shape[0] / height)
    # print(shape, width_step, height_step)
    index = 0
    for i in range(width):
        width_array = img_array[:, i * width_step: (i + 1) * width_step, :]
        for j in range(height):
            height_array = width_array[j *
                                       height_step: (j + 1) * height_step, :, :]
            img = Image.fromarray(height_array.astype('uint8')).convert('RGB')
            img.save(f'{output_path}/{image_name}_{index}.jpg')
            index += 1


def _():
    parent_path = 'source_two'
    output_parent_path = 'image'
    listdir = os.listdir(parent_path)
    for l in listdir:
        images = os.listdir(os.path.join(parent_path, l))
        output_path = os.path.join(output_parent_path, l)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in images:
            image_path = os.path.join(parent_path, l, image)
            image_split_util(image_path, image.split('.')[0], output_path)


def show_image():
    img = image_array('24_13_1_0.jpg')
    # with open('b.txt', 'w') as w:
    #     for im in img:
    #         for ii in im:
    #             w.write(' '.join([str(i) for i in ii])+'==')
    #         w.write('\n')
    img = Image.fromarray(img[0:70, 0:70].astype('uint8')).convert('RGB')
    img.save('demo.jpg')


import cv2 as cv


def contiguous_difference(index):
    index_temp = np.append(index, np.zeros((1, 1)))
    index_temp = np.delete(index_temp, 0).reshape(-1, 1).astype('int')
    temp = np.argwhere(np.delete(index_temp-index, len(index_temp)-1) != 1)
    return temp


def edge_cannny(img):
    # 高斯模糊，降低噪声
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    # 灰度图像
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2BGR)
    # 图片梯度
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # 计算边缘
    # 50和150参数必须符合1：3或者1：2
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)
    # cv.imshow('edge', edge_output)

    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return edge_output


def split(edge_output):
    # print(edge_output.shape)
    # with open('a.txt', 'w') as w:
    #     for e in edge_output:
    #         w.write(' '.join([str(i) for i in e])+'\n')
    sum1 = np.sum(edge_output, axis=0, dtype=np.int)
    sum2 = np.sum(edge_output, axis=1, dtype=np.int)
    # with open('a.txt', 'w') as w:
    #     w.write('\n'.join([str(i) for i in sum1]))
    # index1 = np.argwhere(sum1 == 0)
    # temp1 = contiguous_difference(index1)
    # for i in range(round(len(temp1)/2)+1):
    #     img = Image.fromarray(
    #         edge_output[:, index1[int(temp1[i])][0]: index1[int(temp1[i])+1][0]].astype('uint8'))
    #     img.save(f'{i}_.jpg')

    index2 = np.argwhere(sum2 == 0)
    print(index2)
    temp = contiguous_difference(index2)
    print(temp)
    for i in range(len(temp)):
        print(index2[int(temp[i])][0])
        print(index2[int(temp[i])+1][0])
        print('-----------------------')
        img = Image.fromarray(
            edge_output[index2[int(temp[i])][0]: index2[int(temp[i])+1][0], :].astype('uint8'))
        img.save(f'{i}.jpg')


if __name__ == '__main__':
    # find_edge()
    # show_image()
    # image_edge()
    # sobel()

    # show_image()

    img = cv.imread('image.jpg')
    # print(img.shape)
    img = cv.rotate()
    img = Image.fromarray(
        img[:, :1800, :].astype('uint8'))
    img.save('img.jpg')

    # edge_output = edge_cannny(img)
    # split(edge_output)
