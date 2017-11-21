
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import mxnet as mx
ctx=mx.cpu(0)
import logging
logging.getLogger().setLevel(logging.DEBUG)
from word_utils import *


# In[ ]:


num_hidden=64
embed_size=64
batch_size=100
dataset_size=10000


# In[ ]:


train_set, inverse_train_set, eval_set, inverse_eval_set, max_string_len = generate_train_eval_sets(dataset_size=dataset_size)

train_iter = generate_OH_iterator(train_set=train_set, label_set=inverse_train_set, batch_size=batch_size, max_len=max_string_len)
eval_iter = generate_OH_iterator(train_set=eval_set, label_set=inverse_eval_set, batch_size=batch_size, max_len=max_string_len)


# In[ ]:


data = mx.sym.Variable('data')
label = mx.sym.Variable('softmax_label')

embed = mx.sym.Embedding(
    data=data,
    input_dim=vocab_size_train, # when one hot, lenght of vocabulary; when floats, lenght of array
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
#drop=mx.sym.Dropout(data=fc, p=0.5)
act=mx.sym.Activation(data=fc, act_type='relu')


out = mx.sym.Reshape(data=act, shape=((0,max_string_len,vocab_size_train)))

#out=mx.sym.round(out)

net = mx.sym.LinearRegressionOutput(data=out, label=label)


# In[ ]:


model = mx.module.Module(net)
model.fit(
    train_data=train_iter,
    eval_data=eval_iter,
    eval_metric = 'acc',
    optimizer=mx.optimizer.Adam(rescale_grad=1/batch_size),
    #optimizer_params={'learning_rate':0.001, 'momentum':0.9},
    initializer=mx.initializer.Xavier(),
    batch_end_callback=mx.callback.Speedometer(batch_size, 10),
    num_epoch=8
)


# In[ ]:


import difflib

testset_size=100

test_set, inverse_test_set, _, _, max_len = generate_train_eval_sets(dataset_size=testset_size, max_len=149)
test_iter = generate_iterator(train_set=test_set, label_set=inverse_test_set, batch_size=1)

predictions=model.predict(test_iter)

match_count=0
for i,pred in enumerate(predictions):
    matched = ints2text(onehot2int(mx.ndarray.round(predictions[i]))) == ints2text(inverse_test_set[i])
    if matched:
        match_count+=1
    else:       
        print(i)
        inverse=ints2text(inverse_test_set[i])
        print(inverse)
        inverse_pred=ints2text(onehot2int(mx.ndarray.round(predictions[i])))
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
    

