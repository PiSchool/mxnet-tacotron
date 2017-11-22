# coding: utf-8

# THIS DEMONSTRATES THE CUSTOM ITERATOR WORKS AND EXTRACTS GOOD ONE HOT SAMPLES OF THE FILE PROVIDED

import mxnet as mx
ctx=mx.cpu(0)
import logging
logging.getLogger().setLevel(logging.DEBUG)
from word_utils import *


train_set, inverse_train_set, eval_set, inverse_eval_set, max_len, vocabulary_en, reverse_vocabulary_en = generate_train_eval_sets(desired_dataset_size=50, max_len=0)

vocab_size_train = len(vocabulary_en)
vocab_size_label = len(reverse_vocabulary_en)

train_iter = generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, max_len=max_len, batch_size=10, vocab_size_data=vocab_size_train, vocab_size_label=vocab_size_label)

train_iter.reset()
try:
    while True:
        item= train_iter.next()
        print(item)
        print(item.provide_data)
        print(item.provide_label)
        print(item.data)
        print(item.label)
except StopIteration:
    print("end of iteration")
print("STATS")
print("Train set size:", len(train_set))
print("Eval set size:", len(eval_set))
print("Vocabulary train size:", vocab_size_train)
print("Vocabulary label size:", vocab_size_label)
print("Max words in sentence:", max_len)
