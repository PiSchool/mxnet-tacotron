# coding: utf-8

# THIS DEMONSTRATES THE CUSTOM ITERATOR WORKS AND EXTRACTS GOOD ONE HOT SAMPLES OF THE FILE PROVIDED

import mxnet as mx
ctx=mx.cpu(0)
import logging
logging.getLogger().setLevel(logging.DEBUG)
import word_utils

desired_dataset_size=20000
max_len=10

train_set, inverse_train_set, eval_set, inverse_eval_set, max_string_len, vocabulary_en, reverse_vocabulary_en = word_utils.generate_train_eval_sets(desired_dataset_size=desired_dataset_size, max_len=max_len)

train_set = word_utils.pad_set(train_set, max_string_len)

vocab_size_train = len(vocabulary_en)
vocab_size_label = len(reverse_vocabulary_en)

train_iter = word_utils.generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, max_len=max_string_len, batch_size=1, vocab_size_data=vocab_size_train, vocab_size_label=vocab_size_label)


print("\nSTATS\n-----")
print("Requested data set size:", desired_dataset_size)
print("Actual train set size:", len(train_set))
print("Actual eval set size:", len(eval_set))
print("Vocabulary train size:", vocab_size_train)
print("Vocabulary label size:", vocab_size_label)
print("Desired max words in sentence:", (max_len if max_len > 0 else "unbounded"))
print("Actual max words in sentence:", max_string_len)
exit(0)

try:
    while True:
        item= train_iter.next()
        print(item)
        print(item.provide_data)
        #print([word_utils.onehot2int(sample) for sample in item.data])
        print([word_utils.ints2text(word_utils.onehot2int(sample),reverse_vocabulary_en) for sample in item.data[0]])
        print([word_utils.ints2text(word_utils.onehot2int(sample),reverse_vocabulary_en) for sample in item.data[1]])
        print()
        print(item.provide_label)
        #print(item.label)
        #print([sample for sample in item.label])
        #print([word_utils.onehot2int(sample) for sample in item.label])
        #print([word_utils.ints2text(word_utils.onehot2int(sample),reverse_vocabulary_en) for sample in item.label])
        print()
        print()
except StopIteration:
    print("end of iteration")
