# -*- coding:utf-8 -*-
import jieba
import collections
import os

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


if __name__ == '__main__':
    build_vocabulary()
