# coding: utf-8

import mxnet as mx
import numpy as np
import audio_process
import math
import time
from params import Hyperparams as hp
from queue import Queue
from threading import Thread
from data_loader_multithread import spectrogramsLoader
import multiprocessing
from queue import Empty


class AudioIter(mx.io.DataIter):
    def __init__(self,audiofile_list,
                 data_names, label_names,
                 name= None,
                 num_threads=None,
                 batch_size=10):

        self.max_samples_length = int(hp.max_seconds_length*hp.sr)
        #print("max_samples_length",self.max_samples_length)
        self.num_batches = len(audiofile_list)//batch_size
        self.batch_size = batch_size
        self.cur_batch = 0
        self.hasStarted = False
        self.sleeptime = 10

        self.spectrogramsLoader = spectrogramsLoader(audiofile_list,num_threads)

        self.name = name
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

        #print("size",self.spectrogramsLoader.spectraQueueSize())

        #print("num batches",self.num_batches)

    def __iter__(self):
        return self


    def reset(self):
        print(self.name,":reset")
        self.spectrogramsLoader.reset() if self.hasStarted else self.spectrogramsLoader.start()
        self.hasStarted=True
        self.cur_batch=0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if not self.hasStarted:
            self.spectrogramsLoader.start()
            self.hasStarted=True
        #print(self.name,"cur batch:",self.cur_batch)
        if self.cur_batch < self.num_batches:
            q_size  = self.spectrogramsLoader.spectraQueueSize()
            if q_size<self.batch_size:
                print(self.name,":queue size",q_size)
                print(self.name,"sleep for",self.sleeptime,"seconds")
                time.sleep(self.sleeptime)
            _data=[]
            _labels=[]
            for i in range(self.batch_size):
                spectrograms = self.spectrogramsLoader.get_spectrograms()
                _data.append(np.transpose(spectrograms[0]))
                _labels.append(np.transpose(spectrograms[1]))

            assert len(_data) == self.batch_size
            assert len(_labels) == self.batch_size

            if self.cur_batch < self.num_batches:
                #print(self.name,"add new batch")
                labels = [mx.nd.array(_labels)]#, self.vocab_size_label]

                data = [mx.nd.array(_data)]# self.vocab_size_data)] + label

                self.cur_batch += 1

                return mx.io.DataBatch(
                    data,
                    labels,
                    pad=0,
                    provide_data=self._provide_data,
                    provide_label=self._provide_label
                )

        else:
            raise StopIteration
