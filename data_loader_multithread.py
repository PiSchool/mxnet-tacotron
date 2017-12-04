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
        try:
            while True:
                audio_file = self.audio_queue.get(block = True, timeout = 2)

                lin_path = audio_file+'.lin'
                mel_path = audio_file+'.mel'

                if os.path.exists(lin_path) and os.path.exists(mel_path):
                    lin = pickle.load(open(lin_path, "rb"))
                    mel = pickle.load(open(mel_path, "rb"))
                else:
                    lin, mel = do_spectrograms(audio_file)

                    pickle.dump(lin, open(lin_path, "wb"))
                    pickle.dump(mel, open(mel_path, "wb"))

                self.audio_queue.task_done()
                self.result_queue.put([mel,lin])
        except queue.Empty:
            print("Empty queue")
            pass


class spectrogramsLoader:
    def __init__(self, audio_files, num_threads):
        self.audio_files = audio_files
        self.audio_files_queue = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()

        # Start consumers
        if num_threads > multiprocessing.cpu_count():
            num_threads = multiprocessing.cpu_count()
        self.num_workers = num_threads if num_threads else multiprocessing.cpu_count()
        print('Creating {} consumers'.format(self.num_workers))

    def start(self):
        print("Data loading started")

        for audio_file in (self.audio_files):
            self.audio_files_queue.put(audio_file)

        self.workers = [
            Worker(self.audio_files_queue, self.results)
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()
        # Wait for all of the tasks to finish
        #self.audio_files_queue.join()
    def reset(self):
        print("data loader reset")
        for w in self.workers:
            w.terminate()
            w.join()
        print("All workers terminated")
        print("Create new queues")
        self.audio_files_queue = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        print("Start again")
        self.start()
    def get_spectrograms(self):
        try:
            return self.results.get(block=True, timeout=30)
        except Empty:
            print("Queue get timeout or queue empty")
    def spectraQueueSize(self):
        return self.results.qsize()
