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

from AudioIter import AudioIter

logging.getLogger().setLevel(logging.DEBUG)

import tacotron


if __name__ == "__main__":
    
    np.random.seed(3) #[42 24  3  8  0]
    test_iterator = tacotron.get_test_iterator()

    
    linear_spectrogram = mx.sym.Variable('linear_spectrogram')
    mel_spectrogram = mx.sym.Variable('mel_spectrogram')

    prefix = hp.dataset_name
    checkpoints_dir = expanduser("~")+"/results/CBHG_model/"+prefix+"/1512142976470519"

    sym_1, arg_params_1, aux_params_1 = mx.model.load_checkpoint(checkpoints_dir+"/"+prefix, 1)
    model = mx.mod.Module(symbol=sym_1, context=ctx,data_names=['mel_spectrogram'],label_names=['linear_spectrogram'])

    model.bind(for_training=False, data_shapes = test_iterator.provide_data, label_shapes = test_iterator.provide_label)
    # assign the loaded parameters to the module

    model.set_params(arg_params_1, aux_params_1)

    '''
    Save waveforms of predicted data
    '''
    predictions_1 =model.predict(test_iterator)

    for i,predicted_spectr in enumerate(predictions_1):
        y = audio_process.inv_spectrogram(np.transpose(predicted_spectr.asnumpy()))
        audio_process.save_wave(checkpoints_dir+"/"+prefix+"_checkpoint1_"+str(i),y,hp.sr)
