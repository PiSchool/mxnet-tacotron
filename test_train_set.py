# coding: utf-8

# THIS DEMONSTRATES THE CUSTOM ITERATOR WORKS AND EXTRACTS GOOD ONE HOT SAMPLES OF THE FILE PROVIDED

import mxnet as mx
ctx=mx.cpu(0)
import logging
logging.getLogger().setLevel(logging.DEBUG)
from word_utils import *


train_set, inverse_train_set, eval_set, inverse_eval_set, max_len = generate_train_eval_sets(dataset_size=100)

train_iter = generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, max_len=max_len, batch_size=10)

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
