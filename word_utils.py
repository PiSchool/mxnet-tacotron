# -*- coding: utf-8 -*-
from random import choice, randrange
import mxnet as mx
import numpy as np
import re
import string
from OneHotIterator import OneHotIterator

def read_content(path):
    with open(path) as ins:
        content = ins.read()
    return content

def tokenize(content):
    content = content.lower()
    content = re.sub(r'[^\w\s^<>]',' ',content)
    content = re.sub('http.*? ',' <url> ',content)
    content = re.sub('  ',' ',content)
    return content

def content_to_list(path):
    content = tokenize(read_content(path))
    content = re.sub('  ',' ',content)
    content = content.split('\n')
    return content

def build_vocab(path):
    content = tokenize(read_content(path))
    content = content.replace('\n', ' <eos> ')
    content = re.sub('  ',' ',content)
    content = content.split(' ')
    vocabulary = {}
    idx = 2  # 0 is left for zero-padding
    vocabulary[''] = 0  # put a dummy element here so that len(vocab) is correct
    vocabulary['<sos>'] = 1
    for word in content:
        if len(word) == 0:
            continue
        if not word in vocabulary:
            vocabulary[word] = idx
            idx += 1

    reverse_vocabulary = {vocabulary[word]:word for i,word in enumerate(vocabulary)}
    assert len(reverse_vocabulary) == len(vocabulary)
    return vocabulary, reverse_vocabulary

def prepend_sos_to_list_of_sentences(sentences):
    for i,line in enumerate(sentences):
        sentences[i]="<sos> "+line
    return sentences

def append_eos_to_list_of_sentences(sentences):
    for i,line in enumerate(sentences):
        if len(line) == 0:
            continue
        sentences[i]+=' <eos>'
    return sentences

vocabulary_en, reverse_vocabulary_en = build_vocab('english')
#vocabulary_it, reverse_vocabulary_it = build_vocab('italian')

vocab_size_train = len(vocabulary_en)
vocab_size_label = len(reverse_vocabulary_en)

def ints2text(numbers, reverse_vocabulary):
    return ' '.join([reverse_vocabulary[num] for num in numbers])

def int2onehot(numbers):
    return mx.nd.one_hot(mx.nd.array(numbers),vocab_size)

def onehot2int(matrix):
    fin=[]
    for vec in matrix:
        fin.append(int(vec.argmax(axis=0).asnumpy().tolist()[0]))
    return fin

def text2ints(sentence, vocabulary):
    words = sentence.split(' ')
    words = [vocabulary[w] for w in words if len(w) > 0]
    return words

def generate_train_eval_sets(dataset_size, max_len=0):
    source_list = content_to_list('english')[:dataset_size]
    target_list = [' '.join(sentence.split(' ')[::-1]) for sentence in source_list]

    source_list = append_eos_to_list_of_sentences(source_list)
    target_list = append_eos_to_list_of_sentences(prepend_sos_to_list_of_sentences(target_list))


    source_as_ints = [text2ints(sentence, vocabulary_en) for sentence in source_list]
    target_as_ints = [text2ints(sentence, vocabulary_en) for sentence in target_list]


    if max_len == 0:
        for sentence in source_as_ints:
            max_len = max(max_len, len(sentence))
        for sentence in target_as_ints:
            max_len = max(max_len, len(sentence))


    for i,sentence in enumerate(source_as_ints):
        for _ in range(max_len - len(sentence)):
           sentence.append(0)
        source_as_ints[i] = sentence
    for i,sentence in enumerate(target_as_ints):
        for _ in range(max_len - len(sentence)):
           sentence.append(0)
        target_as_ints[i] = sentence


    eval_indexes = list(set(np.random.randint(0,dataset_size-1, dataset_size//10)))
    train_indexes = np.setdiff1d(np.arange(dataset_size),eval_indexes)

    eval_set = [source_as_ints[i] for i in eval_indexes]
    inverse_eval_set = [target_as_ints[i] for i in eval_indexes]

    train_set = [source_as_ints[i] for i in train_indexes]
    inverse_train_set = [target_as_ints[i] for i in train_indexes]

    return train_set, inverse_train_set, eval_set, inverse_eval_set, max_len
    
def generate_iterator(train_set, label_set, batch_size):
    train_one_hot = mx.nd.one_hot(mx.nd.array(train_set),vocab_size_train)
    label_one_hot = mx.nd.one_hot(mx.nd.array(label_set),vocab_size_label)

    return mx.io.NDArrayIter(
        data=train_one_hot,
        label=label_one_hot,
        batch_size=batch_size
    )

def generate_OH_iterator(train_set, label_set, batch_size, max_len):
    return OneHotIterator(
        data=train_set,
        label=label_set,
        data_names=['data'],
        max_len_data=max_len,
        vocab_size_data=vocab_size_train,
        label_names=['softmax_label'],
        max_len_label=max_len,
        vocab_size_label=vocab_size_label,
        batch_size=batch_size
    )

#train_set, inverse_train_set, eval_set, inverse_eval_set, max_len = generate_train_eval_sets(dataset_size=10000)

#train_iter = generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, max_len=max_len, batch_size=1)

#train_iter.reset()
#item= train_iter.next()
#print(item)
#print(item.data)
