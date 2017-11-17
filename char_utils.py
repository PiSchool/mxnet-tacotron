# -*- coding: utf-8 -*-
from random import choice, randrange
import mxnet as mx

vocabulary=list("abcdef")
EOS='§'
SOS='#'
max_string_len = 15
vocabulary.append(EOS)
vocabulary.append(SOS)
vocab_size=len(vocabulary)
int2char = {i:c for i,c in enumerate(vocabulary)}
char2int = {c:i for i,c in enumerate(vocabulary)}

print(int2char)
print("vocab size: "+str(vocab_size))
print("max string length: "+str(max_string_len))

def generate_strings(min_len, max_len):
    random_length = randrange(min_len, max_len)
    random_char_list = [choice(vocabulary[:-2]) for _ in range(random_length)]
    random_string = ''.join(random_char_list)
    return SOS+random_string+EOS

def text2ints(string):
    return [char2int[char] for char in string]

def ints2text(numbers):
    return ''.join([int2char[num] for num in numbers])

def int2onehot(numbers):
    return mx.nd.one_hot(mx.nd.array(numbers),vocab_size)

def onehot2int(matrix):
    fin=[]
    for vec in matrix:
        fin.append(int(vec.argmax(axis=0).asnumpy().tolist()[0]))
    return fin

string=generate_strings(max_string_len-2, max_string_len-1)
print(string, len(string))
assert ints2text(text2ints(string)) == string

def generate_train_eval_sets(dataset_size):
    train_set = [text2ints(generate_strings(max_string_len-2, max_string_len-1)) for _ in range(dataset_size)]
    inverse_train_set = [[char2int[SOS]]+sentence[1:-1][::-1]+[char2int[EOS]] for sentence in train_set]
    
    eval_set = [text2ints(generate_strings(max_string_len-2, max_string_len-1)) for _ in range(dataset_size//10)]
    inverse_eval_set = [[char2int[SOS]]+sentence[1:-1][::-1]+[char2int[EOS]] for sentence in eval_set]
 
    return train_set, inverse_train_set, eval_set, inverse_eval_set
    
def generate_iterator(train_set, label_set, batch_size):
    return mx.io.NDArrayIter(
        data=mx.nd.one_hot(mx.nd.array(train_set),vocab_size),
    label=mx.nd.one_hot(mx.nd.array(label_set),vocab_size),
    batch_size=batch_size
    )
