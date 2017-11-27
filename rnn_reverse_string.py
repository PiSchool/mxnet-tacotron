# coding: utf-8

# RECURRENT NETWORK THAT REVERSES A SENTENCE

import mxnet as mx
import os.path
import glob
import logging
import pickle
import word_utils

logging.getLogger().setLevel(logging.DEBUG)
ctx=mx.gpu(0)

num_hidden=128
embed_size=256
dataset_size=20000
batch_size = 100
desired_max_len=10 # 0 for unbounded

model_prefix='reverse-string-hemingway'
save = False

# DATASETS AND ITERATORS GENERATION

# TO RESUME A PREVIOUSLY TRAINED MODEL, WE NEED PARAMS, SHAPES AND VOCABULARIES, SINCE THE LATTERS PLAY A ROLE IN THE NETWORK TOPOLOGY

vocabulary_en_pickled_filename = model_prefix+"-vocabulary_en.pickled"
reverse_vocabulary_en_pickled_filename = model_prefix+"-reverse_vocabulary_en.pickled"
max_string_len_pickled_filename = model_prefix+"-max_string_len.pickled"

if os.path.exists(vocabulary_en_pickled_filename) and os.path.exists(reverse_vocabulary_en_pickled_filename) and os.path.exists(max_string_len_pickled_filename) and save:
    max_string_len = pickle.load( open( max_string_len_pickled_filename, "rb" ) )
    vocabulary_en = pickle.load( open( vocabulary_en_pickled_filename, "rb" ) )
    reverse_vocabulary_en = pickle.load( open( reverse_vocabulary_en_pickled_filename, "rb" ) )

    train_set, inverse_train_set, eval_set, inverse_eval_set, _, _, _ = word_utils.generate_train_eval_sets(desired_dataset_size=dataset_size, max_len=desired_max_len)

    # ensure sets fit the size mandated by max_string_len
    train_set = word_utils.pad_set(train_set, max_string_len)
    inverse_train_set = word_utils.pad_set(inverse_train_set, max_string_len)
    eval_set = word_utils.pad_set(eval_set, max_string_len)
    inverse_eval_set = word_utils.pad_set(inverse_eval_set, max_string_len)
else:
    train_set, inverse_train_set, eval_set, inverse_eval_set, max_string_len, vocabulary_en, reverse_vocabulary_en = word_utils.generate_train_eval_sets(desired_dataset_size=dataset_size, max_len=desired_max_len)

vocab_size_train = len(vocabulary_en)
vocab_size_label = len(reverse_vocabulary_en)

print("TRAIN STATS")
print("Train set size:", len(train_set))
print("Eval set size:", len(eval_set))
print("Vocabulary train size:", vocab_size_train)
print("Vocabulary label size:", vocab_size_label)
print("Max words in sentence:", max_string_len)

train_iter = word_utils.generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, max_len=max_string_len, batch_size=batch_size, vocab_size_data=vocab_size_train, vocab_size_label=vocab_size_label)


eval_iter = word_utils.generate_OH_iterator(train_set=eval_set, label_set=inverse_eval_set, batch_size=batch_size, max_len=max_string_len, vocab_size_data=vocab_size_train, vocab_size_label=vocab_size_label)

# NETWORK DEFINITION
source = mx.sym.Variable('source')
target = mx.sym.Variable('target')
label = mx.sym.Variable('softmax_label')

source_embed = mx.sym.Embedding(
    data=source,
    input_dim=vocab_size_train,
    output_dim=embed_size
)
target_embed = mx.sym.Embedding(
    data=target,
    input_dim=vocab_size_label,
    output_dim=embed_size
)

bi_cell = mx.rnn.BidirectionalCell(
    mx.rnn.GRUCell(num_hidden=num_hidden, prefix="gru1_"),
    mx.rnn.GRUCell(num_hidden=num_hidden, prefix="gru2_"),
    output_prefix="bi_"
)

encoder = (bi_cell)
        
_, encoder_state = encoder.unroll(
    length=max_string_len,
    inputs=source_embed,
    merge_outputs=False
)

encoder_state = mx.sym.concat(encoder_state[0][0],encoder_state[1][0])

decoder = mx.rnn.GRUCell(num_hidden=num_hidden*2)

rnn_output, decoder_state = decoder.unroll(
    length=max_string_len,
    begin_state=encoder_state,
    inputs=target_embed,
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
model = mx.module.Module(net, data_names=['source','target'], context=ctx)


max_epoch=8

latest_epoch=0 # dummy, don't change

# search for files matching the pattern of our serialized model
for f in glob.glob(model_prefix+'-*.params'):
    latest_epoch=max(latest_epoch,int(f.split(model_prefix)[1].split('-')[1].split('.')[0]))
latest=str(latest_epoch).zfill(4)

model_params=model_prefix+'-'+latest+'.params'
model_symbols=model_prefix+'-symbol.json'

if os.path.exists(model_params) and os.path.exists(model_symbols) and max_epoch==latest_epoch and save:
    print("Model %s found, loading" % model_prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix,latest_epoch)

    model.bind(
        data_shapes=train_iter.provide_data,
        label_shapes=train_iter.provide_label
    )

    model.set_params(arg_params, aux_params)

    print("Model loaded")

else:
    if save:
        if not os.path.exists(model_params):
            print("File %s not found" % model_params)
        elif not os.path.exists(model_symbols):
            print("File %s not found" % model_symbols)
        elif max_epoch!=latest_epoch:
            print("Epoch mismatch")

        #resume model
        print("Model %s not found, trying checkpoint" % model_prefix)

    latest_epoch=0
    for f in glob.glob(model_prefix+'-*.params'):
        latest_epoch=max(latest_epoch,int(f.split(model_prefix)[1].split('-')[1].split('.')[0]))
    latest=str(latest_epoch).zfill(4)
    latest_checkpoint=model_prefix+'-'+latest+'.params'

    if os.path.exists(latest_checkpoint) and save:
        
        print("Checkpoint %s found, loading" % latest_checkpoint)

        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, latest_epoch)

        print("Checkpoint loaded, resuming training")

        model.fit(
            train_iter,
            eval_data=eval_iter,
            eval_metric='acc',
            optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum':0.9},
            initializer=mx.initializer.Xavier(),
            batch_end_callback = mx.callback.Speedometer(batch_size, 10),
            epoch_end_callback = mx.callback.do_checkpoint(model_prefix),
            arg_params=arg_params,
            aux_params=aux_params,
            begin_epoch=latest_epoch,
            num_epoch=max_epoch
        )
    else:
        #start from zero
        if save:
            print("No checkpoint found for %s, starting from the beginning" % model_prefix)
        else:
            print("Starting from scratch")

        model.fit(
            train_data=train_iter,
            eval_data=eval_iter,
            eval_metric = 'acc',
            optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum':0.9},
            initializer=mx.initializer.Xavier(),
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            epoch_end_callback = mx.callback.do_checkpoint(model_prefix),
            num_epoch=max_epoch
        )

    print("Cleaning up")
    for f in glob.glob(model_prefix+'-*'):
            os.remove(f)

    if save:
        latest=str(max_epoch).zfill(4)
        model_params=model_prefix+'-'+latest+'.params'

        print("Saving model %s" % model_params)
        model.save_params(model_params)
        print("Model saved")

        print("Saving vocabulary")
        pickle.dump(vocabulary_en, open(vocabulary_en_pickled_filename, "wb" ))
        pickle.dump(reverse_vocabulary_en, open(reverse_vocabulary_en_pickled_filename, "wb" ))
        print("Vocabulary saved")

        print("Saving max_string_len")
        pickle.dump(max_string_len, open(max_string_len_pickled_filename, "wb" ))
        print("max_string_len saved")

# TEST WITH UNSEEN DATA (IN THIS CASE IS EVEN SEEN DATA DUE TO THE LOUSY WAY I GENERATE A TEST SET, BUT IT DOESN'T MATER SINCE IT DOESN'T PREDICT ANYTHING)

import difflib

testset_size=10

test_set, inverse_test_set, _, _, _, _, _ = word_utils.generate_train_eval_sets(desired_dataset_size=testset_size, max_len=desired_max_len)

# normalize data to the shapes used during training

test_set = word_utils.pad_set(test_set, max_string_len)
inverse_test_set = word_utils.pad_set(inverse_test_set, max_string_len)

test_iter = word_utils.generate_OH_iterator(train_set=test_set, label_set=inverse_test_set, max_len=max_string_len, batch_size=1, vocab_size_data=vocab_size_train, vocab_size_label=vocab_size_label)

print("TEST STATS")
print("Train set size:", len(test_set))
print("Vocabulary train size:", vocab_size_train)
print("Vocabulary label size:", vocab_size_label)
print("Max words in sentence:", max_string_len)

# PREDICT WITH THE MODEL AND ENJOY THE CRASH. IT DOESN'T HAPPEN WITH batch_size=50, the same used during training

predictions=model.predict(test_iter)

match_count=0
for i,pred in enumerate(predictions):
    matched = word_utils.ints2text(word_utils.onehot2int(predictions[i]), reverse_vocabulary_en) == word_utils.ints2text(inverse_test_set[i], reverse_vocabulary_en)
    if matched:
        match_count+=1
    else:
        print(i)
        inverse=word_utils.ints2text(inverse_test_set[i], reverse_vocabulary_en)
        print(inverse)
        inverse_pred=word_utils.ints2text(word_utils.onehot2int(predictions[i]), reverse_vocabulary_en)
        print(inverse_pred)
        print(matched)
        for i,s in enumerate(difflib.ndiff(inverse, inverse_pred)):
            if s[0]==' ': continue
            elif s[0]=='-':
                print(u'Delete "{}" from position {}'.format(s[-1],i))
            elif s[0]=='+':
                print(u'Add "{}" to position {}'.format(s[-1],i))
        print("--------------------")

print("Matched %d/%d times" % (match_count,testset_size))
