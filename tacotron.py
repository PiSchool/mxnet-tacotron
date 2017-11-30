
# coding: utf-8

# <h1>TACOTRON</h1>

# In[1]:


from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from IPython.display import clear_output
ctx= mx.gpu(0)
import csv
import codecs
import re
import audio_process
import datetime
import os
from os.path import expanduser
import math
import logging
from params import Hyperparams as hp
import time

logging.getLogger().setLevel(logging.DEBUG)


# <h3> DATA SETUP </h3>
#
# <b>Data</b>:
# text - mel spectrograms - linear spectrograms
#
# <b>Shapes</b>:
# (batch_size, pad_to_max_text_length) - (batch_size, 80, pad_to_max_audio_length) - (batch_size, 1025, pad_to_max_audio_length)
#
# <b>Note_1</b>: I'm using a little batch size due my little dummy train dataset <br/>
# <b>Note_2</b>: on Tensorflow implementation there is a reshape step by reduction factor r described in the paper.<br/> Tensorflow data got these shapes:
#
# <b>text</b>:
# (batch_size,length_text)
# <b>mel spectrograms</b>:
# (batch_size, time_frames, 80&ast;r)
# <b>linear spectrograms</b>:
# (batch_size, time_frames, 1025&ast;r)
#
# <br/>
# more info at: https://github.com/Kyubyong/tacotron/blob/master/utils.py#L58

# In[2]:



# THIS IS A VERY SIMPLE ITERATOR THAT, TAKEN A LIST OF SENTENCES ENCODED AS INTEGERS, OUTPUTS THE BATCHES IN ONE HOT FORM (TO OVERCOME MEMORY ALLOCATION ISSUES)

class AudioIter(mx.io.DataIter):
    def __init__(self,
                 audiofile_list,
                 data_names, label_names,
                 batch_size=10):

        self.max_samples_length = int(hp.max_seconds_length*hp.sr)
        #print("max_samples_length",self.max_samples_length)
        self.num_batches = len(audiofile_list)//batch_size
        self.batch_size = batch_size
        self.cur_pointer = 0
        self.cur_batch = 0
        self.audiofile_list = audiofile_list

        max_n_frames = math.ceil(self.max_samples_length/hp.hop_length)
        #print("max_n_frames",max_n_frames)
        self._provide_data = [
            mx.io.DataDesc(
                name=data_name,
                shape=(batch_size, max_n_frames, hp.n_mels),
                layout='NTC') for data_name in data_names
        ]
        self._provide_label = [
            mx.io.DataDesc(
                name=label_name,
                shape=(batch_size, max_n_frames, 1+(hp.n_fft//2)),
                layout='NTC') for label_name in label_names
        ]
        #assert max_len_data == max_len_label
#         self.data = data
#         self.label = label

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0
        self.cur_pointer = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:

            data_batch = []
            label_batch = []
            for i in range(self.batch_size):
                #load the audio file
                wav, sr = audio_process.load_wave(self.audiofile_list[self.cur_pointer])

                assert sr == hp.sr

                wav_length = len(wav)
                diff = self.max_samples_length -wav_length
        #         print("num of zeros to add",diff)
                padded = np.append(wav,np.zeros(diff-1))
                # get the spectrum from the padded sound
                spectrum_lin, spectrum_mel=audio_process.do_spectrograms(y=padded)
        #         print(padded_spectrum_lin.shape)
                # save into the ndarray
        #         spectra_lin[indx,:,:]=np.transpose(spectrum_lin[:,:])
        #         spectra_mel[indx,:,:]=np.transpose(spectrum_mel[:,:])
                data_batch.append(np.transpose(spectrum_mel))
                label_batch.append(np.transpose(spectrum_lin))
                #print(spectrum_lin.shape)
                self.cur_pointer+=1

            label = [mx.nd.array(label_batch)]#, self.vocab_size_label]

            data = [mx.nd.array(data_batch)]# self.vocab_size_data)] + label


            self.cur_batch += 1

            return mx.io.DataBatch(
                data,
                label,
                pad=0,
                provide_data=self._provide_data,
                provide_label=self._provide_label
            )
        else:
            raise StopIteration



# In[3]:


def generate_vocabulary(texts_list):
    # get unique chars and put into a list
    return list(set(''.join(texts_list)))


def generate_chars2numbers_mappings(vocabulary):
    # create a chars <-> numbers mappings
    char2index = {char:i for i,char in enumerate(vocabulary)}
    index2char = {i:char for i,char in enumerate(vocabulary)}

    return char2index,index2char


def text2numbers(texts_list,char2index_mapping):
    numerical_texts=[]
    for text in texts_list:
        numerical_texts.append([char2index_mapping[char] for char in text])
    return numerical_texts

def open_data(input_file_path):

    texts, sound_files = [], []

    reader = csv.reader(codecs.open(input_file_path, 'rb', 'utf-8'))
    for row in reader:
        sound_filename, text, _ = row
        sound_file = hp.sound_fpath +"/"+ sound_filename + ".wav"
        text = re.sub(r"[^ a-z']", "", text.strip().lower())

        texts.append(text)
        sound_files.append(sound_file)

    return texts, sound_files

# In[4]:


def get_iterators():
    texts_list, sound_files_list = open_data(hp.csv_file)

    size=len(sound_files_list)
    # get 10% of dataset as eval data
    eval_indxs = (np.random.randint(0, high=size, size=size//10))

    # remaining indexes for the train
    train_indxs = np.setdiff1d(np.arange(size),eval_indxs)


    print("I will take those for eval:",eval_indxs)
    #print("..and the remaining for train:",train_indxs,"\n")

    train_set = np.ndarray.take(np.asarray(sound_files_list),train_indxs)
    eval_set = np.ndarray.take(np.asarray(sound_files_list),eval_indxs)

    train_iter = AudioIter(train_set,["mel_spectrogram"],["linear_spectrogram"],hp.batch_size)
    eval_iter = AudioIter(eval_set,["mel_spectrogram"],["linear_spectrogram"],hp.batch_size)

    return train_iter, eval_iter


# <h3> Modules </h3>

# <h4> Prenet </h4>

# In[5]:

"""
FC-256-ReLU → Dropout(0.5) → FC-128-ReLU → Dropout(0.5)
"""
def prenet_pass(data):
    fc1 = mx.symbol.FullyConnected(data=data, num_hidden=hp.emb_size, name='prenet_fc1',flatten=False)
    act1 = mx.symbol.Activation(data=fc1, act_type='relu', name='prenet_act1')
    drop1 = mx.symbol.Dropout(act1, p=0.5, name='prenet_drop1')

    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=hp.emb_size//2, name='prenet_fc2', flatten=False)
    act2 = mx.symbol.Activation(data=fc2, act_type='relu', name='prenet_act2')
    prenet_output = mx.symbol.Dropout(act2, p=0.5, name='prenet_drop2')

    return prenet_output


# <h4> Convolution 1D Bank </h4>

# In[6]:


# Convolution bank of K filter
def conv1dBank(conv_input, K): # 1,88,128 # N C W -> 1 80 88
    #(32,88,128)
    #N C W (num_batch, channel, width)

    #The k-th filter got a kernel width of k, with 0<k<=K
    conv=mx.sym.Convolution(data=conv_input, kernel=(1,), num_filter=hp.emb_size//2,name="convBank_1")
    #(32,128,128) ==> (32,K*128,128)
    #(32,num_filter,out_width)

    conv = mx.sym.Activation(data=conv, act_type='relu')
    '''
    BatchNorm: Got error out_grad.size() check failed 1==3 using GPU during fit()
    '''
    #(conv, mean, var) = mx.sym.BatchNorm(data=conv, output_mean_var=True)
    if hp.use_convBank_batchNorm:
        conv = mx.sym.BatchNorm(data=conv, name="batchN_bank_1")

    for k in range(2, K+1):
        in_i = mx.sym.concat(conv_input,mx.sym.zeros((hp.batch_size,hp.emb_size//2,k-1)),dim=2)
        convi = mx.sym.Convolution(data=in_i, kernel=(k,), num_filter=hp.emb_size//2,name="convBank_"+str(k))

        convi = mx.sym.Activation(data=convi,act_type='relu')
        '''
        BatchNorm: Got error out_grad.size() check failed 1==3 using GPU during fit()
        '''

        #(convi, mean, var) = mx.sym.BatchNorm(data=convi, output_mean_var=True)
        if hp.use_convBank_batchNorm:
            convi = mx.sym.BatchNorm(data=convi, name='batchN_bank_'+str(k))
        conv = mx.symbol.concat(conv,convi,dim=1)

    return conv


# <h4> Highway </h4>

# In[7]:


# highway
def highway_layer(data,i=0):
    H= mx.symbol.Activation(
        data=mx.symbol.FullyConnected(data=data, num_hidden=hp.emb_size//2, name="highway_fcH_"+str(i),flatten=False),
        act_type="relu"
    )
    T= mx.symbol.Activation(
        data=mx.symbol.FullyConnected(data=data, num_hidden=hp.emb_size//2, bias=mx.sym.Variable('bias'+str(i),init=mx.initializer.Normal()), name="highway_fcT"+str(i),flatten=False),
        act_type="sigmoid"
    )
    return  H * T + data * (1.0 - T)


# <h4> CBHG </h4>

# In[8]:


# CBHG
def CBHG(data,K,proj1_size,proj2_size,num_unroll):
    bank = conv1dBank(data,K)

    #After the convolutional bank, a max pooling is applied
    #Again here. To obtain always the same dimension I'm padding the input of each operation
    conv_padded = mx.sym.concat(bank,mx.sym.zeros((hp.batch_size,K*(hp.emb_size//2),1)),dim=2)
    poold_bank = mx.sym.Pooling(data=conv_padded, pool_type='max', kernel=(2,), stride=(1,), name="CBHG_pool")
    #(32,1024,127)
    #Now two other projections (convolutions) are done. Same padding thing
    poold_bank_padded = mx.sym.concat(poold_bank,mx.sym.zeros((hp.batch_size,K*(hp.emb_size//2),2)),dim=2)

    proj1 = mx.sym.Convolution(data=poold_bank_padded, kernel=(3,), num_filter=proj1_size, name='CBHG_conv1')
    proj1 = mx.sym.Activation(data=proj1, act_type='relu', name='CBHG_act1')

    if hp.use_proj1_batchNorm:
        proj1 = mx.sym.BatchNorm(data=proj1, name="batchNorm_proj1")

    proj1_padded = mx.sym.concat(proj1,mx.sym.zeros((hp.batch_size,hp.emb_size,2)),dim=2)
    proj2 = mx.sym.Convolution(proj1_padded, kernel=(3,), num_filter=proj2_size, name='CBHG_conv2')

    if hp.use_proj2_batchNorm:
        proj2=mx.sym.BatchNorm(data=proj2, name="batchNorm_proj2")

    #Adding residual connection. The output of the prenet pass is added to proj2
    residual= proj2 + data

    residual = mx.sym.swapaxes(residual,1,2)

    #A 4 highway layers is created
    for i in range(4):
        residual = highway_layer(residual,i)
    highway_pass = residual

    #The highway output is passed to the bidirectional gru cell
    bidirectional_gru_cell = mx.rnn.BidirectionalCell(
        mx.rnn.GRUCell(num_hidden=hp.emb_size//2, prefix='CBHG_gru1'),
        mx.rnn.GRUCell(num_hidden=hp.emb_size//2, prefix='CBHG_gru2'),
        output_prefix='CBHG_bi_'
    )

    bi_gru_outputs, bi_gru_states = bidirectional_gru_cell.unroll(num_unroll, inputs=highway_pass, merge_outputs=True)

    return bi_gru_outputs


# <h4> Encoder </h4>

# In[9]:


# encoder
def encoder(data):
    embed_vector = mx.sym.Embedding(data=data, input_dim=longest_word, output_dim=hp.emb_size, name='encoder_embed')
    prenet_output = prenet_pass(embed_vector)
    return CBHG(prenet_output,16, hp.emb_size//2, hp.emb_size//2)


# <h4> Decoder (stub)</h4>

# In[10]:


# decoder
def decoder(input_spectrogram,context,reduction_factor):
    #embed_vector = mx.sym.Embedding(data=input_spectrogram, input_dim=80, output_dim=emb_size, name='decoder_embed')
    prenet_output = prenet_pass(input_spectrogram)

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


# In[11]:


def postprocess(input_mel_spectrograms,max_audio_length):
    in_cbhg = prenet_pass(input_mel_spectrograms)
    in_cbhg_sw= mx.sym.swapaxes(in_cbhg,1,2)

    bi_gru_out =CBHG(in_cbhg_sw,hp.post_process_num_banks,hp.emb_size,hp.emb_size//2,max_audio_length)

    linear_scale_spectrograms = mx.symbol.FullyConnected(data=bi_gru_out,num_hidden=(hp.n_fft//2)+1,flatten=False)
    return linear_scale_spectrograms


# In[12]:


np.random.seed(3) #[42 24  3  8  0]
traindata_iterator, evaldata_iterator = get_iterators()
linear_spectrogram = mx.sym.Variable('linear_spectrogram')
mel_spectrogram = mx.sym.Variable('mel_spectrogram')


# In[13]:


max_n_frames = math.ceil((hp.max_seconds_length*hp.sr)/hp.hop_length)

print("max_audio_length: ",hp.max_seconds_length)
print("max_n_frames:",max_n_frames)
net = mx.sym.MAERegressionOutput(data=postprocess(mel_spectrogram,max_n_frames), label=linear_spectrogram)
#net = mx.sym.SoftmaxOutput(data=postprocess(mel_spectrogram,max_audio_length), label=linear_spectrogram)
model = mx.mod.Module(symbol=net,
                      context=ctx,
                      data_names=['mel_spectrogram'],
                      label_names=['linear_spectrogram']
                     )
# model.bind(for_training=False, data_shapes= traindata_iterator.provide_data, label_shapes=traindata_iterator.provide_label)
# model.load("/home/stefano/CBHG_model/tacotron_15119700453363569/prefix")


# In[14]:


hp.num_epochs=20

checkpoints_dir = expanduser("~")+"/results/CBHG_model/"+hp.dataset_name+"/"+"".join(str(time.time()).split('.'))
prefix = hp.dataset_name
checkpoint_period = hp.num_epochs//2;
if checkpoint_period < 1:
    checkpoint_period=1

if not os.path.exists(checkpoints_dir):
    print("Creating directory for checkpoints")
    os.makedirs(checkpoints_dir)
print("Start training:")
print("- Dataset name:",hp.dataset_name)
print("- Dataset csv file:",hp.csv_file)
print("- Batch size:",hp.batch_size)
print("- Epochs:",hp.num_epochs)
print("- Checkpoint period:",checkpoint_period)
print("- Conv1DBank. use batch normalization:",hp.use_convBank_batchNorm)
print("- Projection_1. use batch normalization:",hp.use_proj1_batchNorm)
print("- Projection_2. use batch normalization:",hp.use_proj2_batchNorm)

model.fit(
        traindata_iterator,
        eval_data=evaldata_iterator,
        optimizer=mx.optimizer.Adam(rescale_grad=1/hp.batch_size),
        optimizer_params={'learning_rate': 0.0001, 'momentum': 0.9},
        eval_metric='mae',
        batch_end_callback = mx.callback.Speedometer(hp.batch_size, 10),
        epoch_end_callback = mx.callback.do_checkpoint(checkpoints_dir+"/"+prefix,checkpoint_period),
        num_epoch=hp.num_epochs
)


# In[20]:


'''
Load first checkpoint n: num_epochs/2
'''
sym_1, arg_params_1, aux_params_1 = mx.model.load_checkpoint(checkpoints_dir+"/"+prefix, hp.num_epochs//2)
model_1chkpnt = mx.mod.Module(symbol=sym_1, context=ctx,data_names=['mel_spectrogram'],label_names=['linear_spectrogram'])

model_1chkpnt.bind(for_training=False, data_shapes= traindata_iterator.provide_data, label_shapes=traindata_iterator.provide_label)
# assign the loaded parameters to the module

model_1chkpnt.set_params(arg_params_1, aux_params_1)

'''
Save waveforms of predicted data
'''
predictions_1 =model.predict(evaldata_iterator)

for i,predicted_spectr in enumerate(predictions_1):
    y = audio_process.inv_spectrogram(np.transpose(predicted_spectr.asnumpy()))
    audio_process.save_wave(checkpoints_dir+"/"+prefix+"_checkpoint1_"+str(i),y,hp.sr)
#


# In[21]:


'''
Load second checkpoint n: num_epochs
'''
sym_2, arg_params_2, aux_params_2 = mx.model.load_checkpoint(checkpoints_dir+"/"+prefix, hp.num_epochs)
model_2chkpnt = mx.mod.Module(symbol=sym_2, context=ctx,data_names=['mel_spectrogram'],label_names=['linear_spectrogram'])

model_2chkpnt.bind(for_training=False, data_shapes= traindata_iterator.provide_data, label_shapes=traindata_iterator.provide_label)
# assign the loaded parameters to the module

model_2chkpnt.set_params(arg_params_2, aux_params_2)

'''
Save waveforms of predicted data
'''
predictions_2 =model.predict(evaldata_iterator)

for i,predicted_spectr in enumerate(predictions_2):
    y = audio_process.inv_spectrogram(np.transpose(predicted_spectr.asnumpy()))
    audio_process.save_wave(checkpoints_dir+"/"+prefix+"_checkpoint2_"+str(i),y,hp.sr)
#
