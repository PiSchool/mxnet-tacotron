# -*- coding: utf-8 -*-
from random import choice, randrange
import mxnet as mx
import re
import string

def read_content(path):
    with open(path) as ins:
        content = ins.read()
    return content

def tokenize(content):
    content = content.lower()
    content = re.sub(r'[^\w\s]',' ',content)
    content = re.sub('http.*? ',' <url> ',content)
    content = re.sub('  ',' ',content)
    return content

def content_to_list(path):
    content = tokenize(read_content(path))
    content = re.sub('  ',' ',content)
    content = content.split('\n')
    for i,line in enumerate(content):
        if len(line) == 0: 
            continue
        content[i]+='<eos>'
    return content

def build_vocab(path):
    content = tokenize(read_content(path))
    content = content.replace('\n', ' <eos> ')
    content = re.sub('  ',' ',content)
    content = content.split(' ')
    vocabulary = {}
    idx = 2  # 0 is left for zero-padding
    vocabulary[' '] = 0  # put a dummy element here so that len(vocab) is correct
    vocabulary['<sos>'] = 1
    for word in content:
        if len(word) == 0:
            continue
        if not word in vocabulary:
            vocabulary[word] = idx
            idx += 1

    reverse_vocabulary = {vocabulary[word]:word for i,word in enumerate(vocabulary)}
    return vocabulary, reverse_vocabulary

def append_sos(sentences):
    for i,line in enumerate(sentences):
        sentences[i]="<sos>"+line
    return sentences

vocabulary, reverse_vocabulary = build_vocab('english')
vocab_size=len(vocabulary)
print(vocabulary)

def ints2text(numbers):
    return ' '.join([reverse_vocabulary[num] for num in numbers])

def int2onehot(numbers):
    return mx.nd.one_hot(mx.nd.array(numbers),vocab_size)

def onehot2int(matrix):
    fin=[]
    for vec in matrix:
        fin.append(int(vec.argmax(axis=0).asnumpy().tolist()[0]))
    return fin

def text2ints(sentence):
    words = tokenize(sentence).split(' ')
    words = [vocabulary[w] for w in words if len(w) > 0]
    return words

def generate_train_eval_sets(dataset_size):
    source_list = content_to_list('english')[:dataset_size]
    target_list = content_to_list('italian')[:dataset_size]
    target_list = append_sos(target_list)


    source_as_ints = [text2ints(sentence) for sentence in source_list]
    target_as_ints = [text2ints(sentence) for sentence in target_list]

    eval_set = source_as_ints[:dataset_size//10]
    inverse_eval_set = target_as_ints[:dataset_size//10]
    
    train_set = source_as_ints[dataset_size//10:]
    inverse_train_set = target_as_ints[dataset_size//10:]
    
    return train_set, inverse_train_set, eval_set, inverse_eval_set
    
def generate_iterator(train_set, label_set, batch_size):
    return mx.io.NDArrayIter(
        data=mx.nd.one_hot(mx.nd.array(train_set),vocab_size),
    label=mx.nd.one_hot(mx.nd.array(label_set),vocab_size),
    batch_size=batch_size
    )


