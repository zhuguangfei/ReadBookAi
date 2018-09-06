# -*- coding:utf-8 -*-
import re
import jieba
import os


_NUM = '_NUM'


def fenci(training_data):
    seg_list = jieba.cut(training_data)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    return training_ci


def basic_tokenizer(sentence):
    _WORD_SPLIT = "([.,!?\":;)()])"
    _CHWORD_SPLIT = '、|。|，|‘|’'
    str1 = ''
    for i in re.split(_CHWORD_SPLIT, sentence):
        str1 = str1+i
    str2 = ""
    for i in re.split(_WORD_SPLIT, str1):
        str2 = str2+i
    return str2


def get_ch_label(txt_file, Isch=True, normalize_digits=False):
    labels = list()
    labelssz = []
    with open(txt_file, 'rb') as f:
        for label in f:
            linstr1 = label.decode('utf-8')
            if normalize_digits:
                linstr1 = re.sub('\d+', _NUM, linstr1)
            notoken = basic_tokenizer(linstr1)
            if Isch:
                notoken = fenci(notoken)
            else:
                notoken = notoken.split()
            labels.extend(notoken)
            labelssz.append(len(labels))
    return labels, labelssz


def getRawFileList(path):
    files = []
    names = []
    for f in os.listdir(path):
        if not f.endswith('~') or not f == '':
            files.append(os.path.join(path, f))
            names.append(f)
    return files, names


def get_ch_path_text(raw_data_dir, Isch=True, normalize_digits=False):
    pass


def create_vocabulary(vocabulary_file, raw_data_dir, max_vocabulary_size, Isch=True, normalize_digits=True):
    pass
