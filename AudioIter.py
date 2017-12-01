# coding: utf-8

import mxnet as mx
import numpy as np
import audio_process
import math
import multiprocessing
from queue import Empty
from params import Hyperparams as hp
import time




class AudioIter(mx.io.DataIter):
    def __init__(self,
                 audiofile_list,
                 data_names, label_names,
                 batch_size=10):

        self.max_samples_length = int(hp.max_seconds_length*hp.sr)
        print("max_samples_length",self.max_samples_length)
        self.num_batches = len(audiofile_list)//batch_size
        self.batch_size = batch_size
        self.cur_batch = 0
        self.poolsize=multiprocessing.cpu_count()//2
        self.audiofile_list = audiofile_list
        self.batches_queue = multiprocessing.Queue()
        self.files_queue = multiprocessing.Queue()
        for audio_path in self.audiofile_list:
            self.files_queue.put(audio_path)

        self.threadpool = multiprocessing.Pool(1, self.create_batches, (self.files_queue,))

        max_n_frames = math.ceil(self.max_samples_length/hp.hop_length)
        print("max_n_frames",max_n_frames)
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

        #time.sleep(60*2)
        #assert max_len_data == max_len_label
#         self.data = data
#         self.label = label

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

        self.batches_queue = multiprocessing.Queue()
        self.files_queue = multiprocessing.Queue()
        for audio_path in self.audiofile_list:
            self.files_queue.put(audio_path)

        self.threadpool = multiprocessing.Pool(1, self.create_batches, (self.files_queue,))

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        try:
            if self.cur_batch < (self.num_batches - self.poolsize):

                batch = self.batches_queue.get(block = True,timeout = 10)
                data_batch = batch[0]
                label_batch = batch[1]

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
                self.threadpool.terminate()
                raise StopIteration
        except Empty:
            self.threadpool.terminate()
            raise StopIteration
        #except Exception as e:
        #    raise StopIteration

    def create_batches(self,queue):

        cur_pointer = 0
        data_batch = []
        label_batch = []
        try:
            while True:
                audiofile = queue.get(block=True, timeout = 30)

                #load the audio file
                wav, sr = audio_process.load_wave(audiofile)

                assert sr == hp.sr

                wav_length = len(wav)
                diff = self.max_samples_length -wav_length
                #print("num of zeros to add",diff)
                zeros = np.zeros(diff-1) if (diff-1)>0 else []
                #print("zeros len:",len(zeros))
                #print("wav len:",len(wav))
                #print("wav shape:",wav.shape)
                padded = np.append(wav,zeros)
                #to be totally sure
                padded= padded[0:self.max_samples_length]
                #print("wav pad:",len(padded))
                # get the spectrum from the padded sound
                spectrum_lin, spectrum_mel=audio_process.do_spectrograms(y=padded)
                #print(spectrum_mel.shape)
                # save into the ndarray
                # spectra_lin[indx,:,:]=np.transpose(spectrum_lin[:,:])
                # spectra_mel[indx,:,:]=np.transpose(spectrum_mel[:,:])
                data_batch.append(np.transpose(spectrum_mel))
                label_batch.append(np.transpose(spectrum_lin))
                #print(spectrum_lin.shape)
                cur_pointer+=1

                if cur_pointer == self.batch_size :
                    self.batches_queue.put([data_batch,label_batch], block=True)
                    cur_pointer = 0
                    data_batch = []
                    label_batch = []
        except Empty:
            data_batch[:]
            label_batch[:]
        #except Exception as e:
        #    print(e)
