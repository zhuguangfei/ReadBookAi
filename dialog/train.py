# -*- coding:utf-8 -*-
import os

import datautil as dutil
from translate import datautil

checkpoint_dir = 'datacn/checkpoints/'

_buckets = [(5, 5), (10, 10), (20, 20)]


def getdialogInfo():
    vocabch, rev_vocabch = dutil.initialize_vocabulary(
        os.path.join(dutil.data_dir, dutil.vocabulary_filech))
    vocab_sizech = len(vocabch)
    filesfrom, _ = dutil.getRawFileList(datautil.data_dir+'fromids/')
    filesto, _ = dutil.getRawFileList(datautil.data_dir+'toids/')
    source_train_file_path = filesfrom[0]
    target_train_file_path = filesto[0]
    return vocab_sizech, vocab_sizech, rev_vocabch, rev_vocabch, source_train_file_path, target_train_file_path


def main():
    vocab_sizeen, vocab_sizech, rev_vocaben, rev_vocabch, source_train_file_path, target_train_file_path = getdialogInfo()
