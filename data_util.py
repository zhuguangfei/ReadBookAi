# -*- coding:utf-8 -*-
import collections
import os
import shutil

import jieba
import numpy as np
from PIL import Image

import cv2 as cv
import image_util
from image_util import edge_cannny

_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_UNK = '_UNK'

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def build_vocabulary():
    words = []
    labels_len = []
    labels = os.listdir('labels')
    for label in labels:
        with open(os.path.join('labels', label), 'r', encoding='utf-8') as r:
            label_words = []
            lines = r.read().split('\n')
            for line in lines:
                words = words + [word for word in line]
                label_words = label_words + [word for word in line]
            labels_len.append(len(label_words))
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    count.extend(collections.Counter(words).most_common(len(words) - 1))
    with open('words.txt', 'w', encoding='utf-8') as w:
        w.write(f'{np.max(labels_len)}\n\n')
    with open('words.txt', 'a', encoding='utf-8') as w:
        for c in count:
            w.write(f'{c[0]}\n')


def get_vocabulary():
    with open('words.txt', 'r', encoding='utf-8') as r:
        words = r.read().strip().split('\n')
    words_index = dict()
    for k, v in enumerate(words):
        if k > 0:
            words_index[v] = k - 1
    return words_index


def gen_label():
    words_index = get_vocabulary()
    labels = os.listdir('labels')
    for label in labels:
        with open(
            os.path.join('labelsids', label.split('.')[0] + '_label.txt'),
            'w',
            encoding='utf-8',
        ) as w:
            indexs = []
            with open(os.path.join('labels', label), 'r', encoding='utf-8') as r:
                line = r.read()
                for word in line:
                    if word == '\n':
                        indexs.append(str(words_index.get('_EOS')))
                    else:
                        indexs.append(str(words_index.get(word)))
            if len(indexs) > 0:
                w.write(' '.join(indexs))


def move_image():
    labels = os.listdir('source_two')
    for label in labels:
        if label.startswith('text'):
            path = os.path.join('source_two', label)
            ll = os.listdir(path)
            for file_path in ll:
                shutil.move(
                    os.path.join('source_two', label,
                                 file_path), f'labels/{file_path}'
                )
                shutil.move(
                    os.path.join(
                        'source_two',
                        label.replace('text', 'image'),
                        file_path.replace('.txt', '.jpg'),
                    ),
                    f"images/{file_path.replace('.txt', '.jpg')}",
                )


def normal_image():
    images = os.listdir('images')
    for image in images:
        image_path = os.path.join('images', image)
        img = cv.imread(image_path)
        img = edge_cannny(img)
        img = Image.fromarray(img.astype('uint8'))
        img.save(os.path.join('normal_image', image))


def label_length():
    labels = os.listdir('labels')
    count = []
    for label in labels:
        with open(os.path.join('labels', label), 'r', encoding='utf-8') as r:
            count.append(len(r.read().split(' ')))
    count = collections.Counter(count).most_common(len(count) - 1)
    print(count[0][1])


max_len = 35


def get_xy():
    words_size = len(get_vocabulary().keys())
    images = os.listdir('normal_image')
    X = []
    for image in images:
        image = Image.open(os.path.join('normal_image', image))
        X.append(np.array(image))
    labels = os.listdir('labelsids')
    Y = []
    for label in labels:
        with open(os.path.join('labelsids', label), 'r') as r:
            words = r.read().split(' ')
            words = [word for word in words if word.strip()]
            if len(words) <= max_len:
                for _ in range(max_len - len(words)):
                    words.append(str('0'))
            else:
                words = words[0:max_len]
            y = []
            for w in words:
                word_label = np.zeros([words_size], dtype=np.int64)
                word_label[int(w)] = 1
                y.append(word_label)
            Y.append(y)
    X = np.array(X, np.float64)
    Y = np.array(Y, np.flaot64)
    return X, Y


def get_xy_test():
    images = os.listdir('image_test')
    X = []
    for image in images:
        image = Image.open(os.path.join('image_test', image))
        X.append(np.array(image))
    labels = os.listdir('label_test')
    Y = []
    for label in labels:
        with open(os.path.join('label_test', label), 'r') as r:
            words = r.read().split(' ')
            words = [word for word in words if word.strip()]
            if len(words) <= max_len:
                for _ in range(max_len - len(words)):
                    words.append(str('0'))
            else:
                words = words[0:max_len]
            y = []
            for w in words:
                y.append(int(w))
            Y.append(y)
    X = np.array(X, np.int16)
    Y = np.array(Y, np.int16)
    return X, Y


if __name__ == '__main__':
    # build_vocabulary()
    # gen_label()
    move_image()
    # label_length()
    # get_xy()
    # normal_image()
    # word_label = np.zeros([10, 1], dtype=np.int64)
    # word_label[2] = 1
    # print(word_label.shape)
    # pass
