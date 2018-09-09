# -*- coding:utf-8 -*-
from translate import datautil
import os
hidden_size = 100
checkpoint_dir = 'datacn/checkpoints/'
_buckets = [(5, 5), (10, 10), (20, 20)]


def getdialogInfo():
    vocabch, rev_vocabch = datautil.initialize_vocabulary(
        os.path.join(datautil.data_dir, datautil.vocabulary_filech))
    vocab_sizech = len(vocabch)
    filesfrom, _ = datautil.getRawFileList(datautil.data_dir+'fromids/')
    filesto, _ = datautil.getRawFileList(datautil.data_dir+'toids/')
    source_train_file_path = filesfrom[0]
    target_train_file_path = filesto[0]
    return vocab_sizech, vocab_sizech, vocabch, rev_vocabch


def main():
    vocab_sizeen, vocab_sizech, vocab_sizeen, rev_vocabch = getdialogInfo()
    conversation_history = []
    while True:
        prompt = '请输入：'
        sentence = input(prompt)
        conversation_history.append(sentence)
        conversation_history = conversation_history[-convo_hist_limit:]
        token_ids = list(reversed(datautil.sentence_to_ids(
            " ".join(conversation_history), vocaben, normalize_digits=True, Isch=True)))
