
import mxnet as mx
import numpy as np
from params import Hyperparams as hp

"""
FC-256-ReLU → Dropout(0.5) → FC-128-ReLU → Dropout(0.5)
"""
def prenet_pass(data):
    fc1 = mx.symbol.FullyConnected(data=data, num_hidden=hp.embed_size, name='prenet_fc1')
    act1 = mx.symbol.Activation(data=fc1, act_type='relu', name='prenet_act1')
    drop1 = mx.symbol.Dropout(act1, p=0.5, name='prenet_drop1')

    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=hp.emb_size//2, name='prenet_fc2')
    act2 = mx.symbol.Activation(data=fc2, act_type='relu', name='prenet_act2')
    prenet_output = mx.symbol.Dropout(act2, p=0.5, name='prenet_drop2')

    return prenet_output

# banco di filtri convolutivi. Vengono creati K filtri con kernel 1D di dimensione:k
def conv1dBank(conv_input, K):
    conv=mx.sym.Convolution(data=conv_input, kernel=(1,1), num_filter=hp.emb_size//2)
    (conv, mean, var) = mx.sym.BatchNorm(data=conv, output_mean_var=True)
    conv = mx.sym.Activation(data=conv, act_type='relu')
    for k in range(2, K+1):
        convi = mx.sym.Convolution(data=conv_input, kernel=(k,1), num_filter=hp.emb_size//2)
        (convi, mean, var) = mx.sym.BatchNorm(data=convi, output_mean_var=True)
        convi = mx.sym.Activation(data=convi, act_type='relu')
        conv = mx.symbol.concat(conv,convi)
    return conv
# highway
def highway_layer(data):
    H= mx.symbol.Activation(
        data=mx.symbol.FullyConnected(data=data, num_hidden=hp.emb_size//2, name="highway_fcH"),
        act_type="relu"
    )
    T= mx.symbol.Activation(
        data=mx.symbol.FullyConnected(data=data, num_hidden=hp.emb_size//2, bias=mx.sym.Variable('bias'), name="highway_fcT"),
        act_type="sigmoid"
    )
    return  H * T + data * (1.0 - T)

# CBHG
def CBHG(data,K,proj1_size,proj2_size):
    #se si usa infer_shape su convbank dando la dimensione dell'input, viene dedotta la shape appunto
    bank = conv1dBank(data,K)
    poold_bank = mx.sym.Pooling(data=bank, pool_type='max', kernel=(2, 1), stride=(1,1), name="CBHG_pool")

    proj1 = mx.sym.Convolution(data=poold_bank, kernel=(3,1), num_filter=proj1_size, name='CBHG_conv1')
    (proj1, proj1_mean, proj1_var) = mx.sym.BatchNorm(data=proj1, output_mean_var=True, name='CBHG_batch1')
    proj1 = mx.sym.Activation(data=proj1, act_type='relu', name='CBHG_act1')

    proj2 = mx.sym.Convolution(proj1, kernel=(3,1), num_filter=proj2_size, name='CBHG_conv2')
    (proj2, proj2_mean, proj2_var) = mx.sym.BatchNorm(data=proj2, output_mean_var=True, name='CBHG_batch2')

    residual= proj2 + data

    for i in range(4):
        residual = highway_layer(residual)
    highway_pass = residual

    bidirectional_gru_cell = mx.rnn.BidirectionalCell(
        mx.rnn.GRUCell(num_hidden=hp.emb_size//2, prefix='CBHG_gru1'),
        mx.rnn.GRUCell(num_hidden=hp.emb_size//2, prefix='CBHG_gru2')
    )
    outputs, states = bidirectional_gru_cell.unroll(1, inputs=highway_pass, merge_outputs=True)
    return outputs

# encoder
def encoder(data, longest_word):
    embed_vector = mx.sym.Embedding(data=data, input_dim=longest_word, output_dim=hp.emb_size, name='encoder_embed')
    prenet_output = prenet_pass(embed_vector)
    return CBHG(prenet_output,16, hp.emb_size//2, hp.emb_size//2)

# decoder
def decoder(input_spectrogram,context,reduction_factor):
    embed_vector = mx.sym.Embedding(data=input_spectrogram, input_dim=80, output_dim=hp.emb_size, name='decoder_embed')
    prenet_output = prenet_pass(embed_vector)

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.GRUCell(num_hidden=hp.emb_size,prefix='decoder_layer1_'))
    stack.add(mx.rnn.GRUCell(num_hidden=hp.emb_size,prefix='decoder_layer2_'))

    residual_gru_stack = mx.rnn.ResidualCell(stack)

    gru_outputs,states = residual_gru_stack.unroll(length=1,
                                               inputs=prenet_output,
                                               begin_state=context,
                                               merge_outputs=True)

    predicted_frames = mx.symbol.Activation(
        data=mx.symbol.FullyConnected(data=gru_outputs, num_hidden=80*reduction_factor),
        act_type="relu"
    )

    return predicted_frames, states
