# -*- coding:utf-8 -*-
import collections
import os
import re
import jieba
from random import shuffle
import numpy as np
import tensorflow.gfile as gfile

_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_UNK = '_UNK'

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_NUM = '_NUM'

data_dir = 'fanyichina/'
raw_data_dir = 'fanyichina/yuliao/from'
raw_data_dir_to = 'fanyichina/yuliao/to'
vocabulary_fileen = 'dicten.txt'
vocabulary_filech = 'dictch.txt'

plot_histograms = plot_catter = True
vocab_size = 4000

max_num_lines = 1
max_target_size = 200
max_source_size = 200


def getRawFileList(path):
    files = []
    names = []
    for f in os.listdir(path):
        if not f.endswith('~') or not f == "":
            files.append(os.path.join(path, f))
            names.append(f)
    return files, names


def basic_tokenizer(sentence):
    _WORD_SPLIT = "([.,!?\"':;])"
    _CHWORD_SPLIT = '、|。|，|’|‘'
    str1 = ""
    for i in re.split(_CHWORD_SPLIT, sentence):
        str1 = str1 + i
    str2 = ""
    for i in re.split(_WORD_SPLIT, str1):
        str2 = str2 + i
    return str2


def get_ch_lable(txt_file, Isch=True, normalize_digits=False):
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


def get_ch_path_text(raw_data_dir, Isch=True, normalize_digits=False):
    text_files, _ = getRawFileList(raw_data_dir)
    labels = []
    training_dataszs = list([0])
    if len(text_files) == 0:
        return labels
    shuffle(text_files)
    for text_file in text_files:
        training_data, training_datasz = get_ch_lable(text_file, Isch, normalize_digits)
        training_ci = np.array(training_data)
        training_ci = np.reshape(training_ci, [-1])
        labels.append(training_ci)

        training_datasz = np.array(training_datasz) + training_dataszs[-1]
        training_dataszs.extend(list(training_datasz))
    return labels, training_dataszs


def sentence_to_ids(sentence, vocabulary, normalize_digits=True, Isch=True):
    if normalize_digits:
        sentence = re.sub('\d+', _NUM, sentence)
    notoken = basic_tokenizer(sentence)
    if Isch:
        notoken = fenci(notoken)
    else:
        notoken = notoken.split()
    idsdata = [vocabulary.get(w, UNK_ID) for w in notoken]
    return idsdata


def textfile_to_idsfile(
    data_file_name, target_file_name, vocab, normalize_digits=True, Isch=True
):
    if not gfile.Exists(target_file_name):
        with gfile.GFile(data_file_name, mode='rb') as data_file:
            with gfile.GFile(target_file_name, mode='w') as ids_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print()
                    token_ids = sentence_to_ids(line, vocab, normalize_digits, Isch)
                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')


def ids2texts(indices, rev_vocab):
    texts = []
    for index in indices:
        texts.append(rev_vocab[index])
    return texts


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode='r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError()


def textdir_to_idsdir(textdir, idsdir, vocab, normalize_digits=True, Isch=True):
    text_files, filenames = getRawFileList(textdir)
    if len(text_files) == 0:
        raise ValueError()

    for text_file, name in zip(text_files, filenames):
        textfile_to_idsfile(text_file, idsdir + name, vocab, normalize_digits, Isch)


def fenci(training_data):
    seg_list = jieba.cut(training_data)
    training_ci = ' '.join(seg_list)
    training_ci = training_ci.split()
    return training_ci


def create_vocabulary(
    vocabulary_file, raw_data_dir, max_vocabulary_size, Isch=True, normalize_digits=True
):
    texts, textssz = get_ch_lable(raw_data_dir, Isch, normalize_digits)


def main():
    vocabulary_filenameen = os.path.join(data_dir, vocabulary_fileen)
    vocabulary_filenamech = os.path.join(data_dir, vocabulary_filech)
