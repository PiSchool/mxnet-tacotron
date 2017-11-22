# -*- coding: utf-8 -*-

# A LOT OF UTILITIES TO READING A TEXT, EXTRACTING THE VOCABULARY, SPLITTING INTO DATASET/EVALSET, CONVERTING TO INTEGER OR ONE-HOT REPRESENTATION AND BACK

from random import choice, randrange
import mxnet as mx
import numpy as np
import re
import string
from OneHotIterator import OneHotIterator


def read_content(path, dataset_size=10000):
    content = ''
    i = 0
    with open(path) as text:
        for line in text:
            i += 1
            if i == dataset_size:
                break
            content += line
    return content

def tokenize(content):
    content = content.lower()
    content = re.sub(r'[^\w\s^<>]',' ',content)
    content = re.sub('http.*? ',' <url> ',content)
    content = re.sub('  ',' ',content)
    return content

def content_to_list(path, dataset_size=10000):
    content = tokenize(read_content(path, dataset_size))
    content = re.sub('  ',' ',content)
    content = content.split('\n')
    stripped_content = []
    for sentence in content:
        if len(sentence) > 0:
            stripped_content.append(sentence.strip(' '))
    return stripped_content

def build_vocab(corpus_as_list_of_sentences):
    vocabulary = {}
    vocabulary[''] = 0  # put a dummy element here so that len(vocab) is correct
    vocabulary['<sos>'] = 1
    vocabulary['<eos>'] = 2
    idx = 3  # start from here
    for sentence in corpus_as_list_of_sentences:
        for word in sentence.split(' '):
            if len(word) > 0 and not word in vocabulary:
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

def generate_train_eval_sets(desired_dataset_size, path='english', max_len=0):
    source_list = content_to_list(path, desired_dataset_size)

    if max_len > 0:
        # prune length of sentences
        for i,sentence in enumerate(source_list):
            words = sentence.split(' ')
            if len(words) >= max_len:
                source_list[i] = ' '.join(words[:max_len-1])

    actual_dataset_size = len(source_list)

    target_list = [' '.join(sentence.split(' ')[::-1][:-1]) for sentence in source_list]

    vocabulary_en, reverse_vocabulary_en = build_vocab(corpus_as_list_of_sentences=source_list)
 
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
        diff = max_len - len(sentence)
        if diff > 0:
            for _ in range(diff):
                sentence.append(0)
            source_as_ints[i] = sentence

    for i,sentence in enumerate(target_as_ints):
        diff = max_len - len(sentence)
        if diff > 0:
            for _ in range(diff):
                sentence.append(0)
            target_as_ints[i] = sentence

    """
    for row in source_as_ints:
        print(len(row),row)
    for row in target_as_ints:
        print(len(row),row)
    exit(0)
    """

    eval_indexes = list(set(np.random.randint(0,actual_dataset_size-1, actual_dataset_size//10)))
    train_indexes = np.setdiff1d(np.arange(actual_dataset_size),eval_indexes)

    eval_set = [source_as_ints[i] for i in eval_indexes]
    inverse_eval_set = [target_as_ints[i] for i in eval_indexes]

    train_set = [source_as_ints[i] for i in train_indexes]
    inverse_train_set = [target_as_ints[i] for i in train_indexes]

    return train_set, inverse_train_set, eval_set, inverse_eval_set, max_len, vocabulary_en, reverse_vocabulary_en
    
def generate_OH_iterator(train_set, label_set, batch_size, max_len, vocab_size_data, vocab_size_label):
    return OneHotIterator(
        data=train_set,
        label=label_set,
        data_names='data',
        max_len_data=max_len,
        vocab_size_data=vocab_size_data,
        label_names='softmax_label',
        max_len_label=max_len,
        vocab_size_label=vocab_size_label,
        batch_size=batch_size
    )
