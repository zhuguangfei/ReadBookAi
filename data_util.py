# -*- coding:utf-8 -*-
import jieba
import collections
import os, shutil

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
    for i in range(12):
        file_paths = os.listdir(f'text{i+1}')
        for file_path in file_paths:
            with open(
                os.path.join(f'text{i+1}', file_path), 'r', encoding='utf-8'
            ) as r:
                lines = r.read().split('\n')
                for line in lines:
                    words = words + [word for word in line]
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    count.extend(collections.Counter(words).most_common(len(words) - 1))
    with open('words.txt', 'w', encoding='utf-8') as w:
        for c in count:
            w.write(f'{c[0]}\n')


def get_vocabulary():
    with open('words.txt', 'r', encoding='utf-8') as r:
        words = r.read().split('\n')
    words_index = dict()
    for k, v in enumerate(words):
        words_index[v] = k
    return words_index


def gen_label():
    words_index = get_vocabulary()
    for i in range(12):
        file_paths = os.listdir(f'text{i+1}')
        for file_path in file_paths:
            if file_path != 'label':
                with open(
                    os.path.join(f'label', file_path.split('.')[0] + '_label.txt'),
                    'w',
                    encoding='utf-8',
                ) as w:
                    indexs = []
                    with open(
                        os.path.join(f'text{i+1}', file_path), 'r', encoding='utf-8'
                    ) as r:
                        line = r.read()
                        for word in line:
                            if word == '\n':
                                indexs.append(str(words_index.get('_EOS')))
                            else:
                                indexs.append(str(words_index.get(word)))
                    if len(indexs) > 0:
                        w.write(' '.join(indexs))


def move_image():
    labels = os.listdir('label')
    images = []
    for label in labels:
        images.append(label.replace('_label.txt', '.jpg'))
    # print(images)
    for i in range(12):
        file_paths = os.listdir(f'image{i+1}')
        for file_path in file_paths:
            if file_path in images:
                shutil.copyfile(
                    os.path.join(f'image{i+1}', file_path), f'image/{file_path}'
                )


def label_length():
    labels = os.listdir('label')
    count = []
    for label in labels:
        with open(os.path.join('label', label), 'r', encoding='utf-8') as r:
            count.append(len(r.read().split(' ')))
    count = collections.Counter(count).most_common(len(count) - 1)
    print(count[0][1])


from PIL import Image
import numpy as np


def get_xy():
    images = os.listdir('image')
    X = []
    for image in images:
        image = Image.open(os.path.join('image', image))
        X.append(np.array(image))
    labels = os.listdir('label')
    Y = []
    for label in labels:
        with open(os.path.join('label', label), 'r') as r:
            words = r.read().split(' ')
            words = [word for word in words if word.strip()]
            if len(words) <= 80:
                for _ in range(80 - len(words)):
                    words.append(str('0'))
            else:
                words = words[0:80]
            y = []
            for w in words:
                y.append(int(w))
            Y.append(y)
    X = np.array(X, np.int16)
    Y = np.array(Y, np.int16)
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
            if len(words) <= 80:
                for _ in range(80 - len(words)):
                    words.append(str('0'))
            else:
                words = words[0:80]
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
    # move_image()
    # label_length()
    get_xy()
