# coding: utf-8

# THIS IS A BASIC TESTCASE OF MXNET FAILING IN READING THE DATA FED BY THE CUSTOM ONE HOT ITERATOR

import mxnet as mx
ctx=mx.cpu(0)
import logging
logging.getLogger().setLevel(logging.DEBUG)
from word_utils import *


num_hidden=64
embed_size=64
batch_size=50
dataset_size=5000

# DATASETS AND ITERATORS GENERATION
train_set, inverse_train_set, eval_set, inverse_eval_set, max_string_len = generate_train_eval_sets(dataset_size=dataset_size)

train_iter = generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, batch_size=batch_size, max_len=max_string_len)
eval_iter = generate_OH_iterator(train_set=eval_set, label_set=inverse_eval_set, batch_size=batch_size, max_len=max_string_len)

# NETWORK DEFINITION
data = mx.sym.Variable('data')
label = mx.sym.Variable('softmax_label')

embed = mx.sym.Embedding(
    data=data,
    input_dim=vocab_size_train, 
    output_dim=embed_size
)

bi_cell = mx.rnn.BidirectionalCell(
    mx.rnn.GRUCell(num_hidden=num_hidden, prefix="gru1_"),
    mx.rnn.GRUCell(num_hidden=num_hidden, prefix="gru2_"),
    output_prefix="bi_"
)

encoder = mx.rnn.ResidualCell(bi_cell)
        
_, encoder_state = encoder.unroll(
    length=max_string_len,
    inputs=embed,
    merge_outputs=False
)

encoder_state = mx.sym.concat(encoder_state[0][0],encoder_state[1][0])

decoder = mx.rnn.GRUCell(num_hidden=num_hidden*2)

rnn_output, decoder_state = decoder.unroll(
    length=num_hidden*2,
    inputs=encoder_state,
    merge_outputs=True
)

flat=mx.sym.Flatten(data=rnn_output)

fc=mx.sym.FullyConnected(
    data=flat,
    num_hidden=max_string_len*vocab_size_train
)
act=mx.sym.Activation(data=fc, act_type='relu')


out = mx.sym.Reshape(data=act, shape=((0,max_string_len,vocab_size_train)))

net = mx.sym.LinearRegressionOutput(data=out, label=label)

# FIT THE MODEL
model = mx.module.Module(net)

# THIS RESULTS IN A BAD ALLOCATION AND A HASTY CORE DUMP..
model.fit(
    train_data=train_iter,
    eval_data=eval_iter,
    eval_metric = 'acc',
    optimizer=mx.optimizer.Adam(rescale_grad=1/batch_size),
    initializer=mx.initializer.Xavier(),
    batch_end_callback=mx.callback.Speedometer(batch_size, 10),
    num_epoch=8
)
