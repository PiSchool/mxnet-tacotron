import numpy as np
import multiprocessing
from queue import Empty
import time
import os.path
import audio_process
import pickle
from params import Hyperparams as hp

def do_spectrograms(audio_file):
    wav, sr = audio_process.load_wave(audio_file)
    max_samples_length = int(hp.max_seconds_length*hp.sr)
    assert sr == hp.sr
    diff = max_samples_length - len(wav)
    #print("num of zeros to add",diff)
    zeros = np.zeros(diff-1) if (diff-1)>0 else []
    #print("zeros len:",len(zeros))
    #print("wav len:",len(wav))
    #print("wav shape:",wav.shape)
    padded = np.append(wav,zeros)
    #to be totally sure
    padded= padded[0:max_samples_length]

    return audio_process.do_spectrograms(y=padded)


class Worker(multiprocessing.Process):

    def __init__(self, audio_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.audio_queue = audio_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        print("got", proc_name, self.pid, len(self.audio_queue), flush=True)

        for audio_file in self.audio_queue :

            lin_path = audio_file+'.lin'
            mel_path = audio_file+'.mel'

            if os.path.exists(lin_path) and os.path.exists(mel_path):
                lin = pickle.load(open(lin_path, "rb"))
                mel = pickle.load(open(mel_path, "rb"))
            else:
                lin, mel = do_spectrograms(audio_file)

                pickle.dump(lin, open(lin_path, "wb"))
                pickle.dump(mel, open(mel_path, "wb"))

            self.result_queue.put([mel,lin])

        print("Empty queue",self.name,flush=True)


class spectrogramsLoader:
    def __init__(self, audio_files, num_threads):
        self.audio_files = audio_files
        self.results = multiprocessing.Queue()

        # Start consumers
        if num_threads > multiprocessing.cpu_count():
            num_threads = multiprocessing.cpu_count()
        self.num_workers = num_threads if num_threads else multiprocessing.cpu_count()

        self.workers_queues = [[] for _ in range(self.num_workers)]
        for i, audio_file in enumerate(self.audio_files):
            self.workers_queues[i % self.num_workers].append(audio_file)


        print('Creating {} consumers'.format(self.num_workers))

    def start(self):
        print("Data loading started")


        self.workers = []
        print("num workers",self.num_workers,flush=True)
        for i in range(self.num_workers):
            w=Worker(self.workers_queues[i], self.results)
            w.start()
            self.workers.append(w)
            print(w.name,"started",flush=True)

    def reset(self):
        print("data loader reset",flush=True)
        for w in self.workers:
            w.terminate()
            print(w.name,"terminate",flush=True)
            w.join()
            print(w.name,"joined",flush=True)

        print("All workers terminated",flush=True)
        print("Create new queues")
        self.results = multiprocessing.Queue()
        print("Start again")
        self.start()
    def get_spectrograms(self):
        try:
            return self.results.get(block=True, timeout=30)
        except Empty:
            print("Queue get timeout or queue empty")
            pass
    def spectraQueueSize(self):
        return self.results.qsize()
