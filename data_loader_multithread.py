import numpy as np
import multiprocessing
from queue import Empty
import time
import audio_process
from params import Hyperparams as hp

def do_spectrograms(audioFile):
    wav, sr = audio_process.load_wave(audioFile)
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
        while True:
            audioFile = self.audio_queue.get()
            if audioFile is None:
                # Poison pill means shutdown
                print('{}: Exiting'.format(proc_name))
                self.audio_queue.task_done()
                break
            #print('{}: {}'.format(proc_name, audioFile))
            lin, mel = do_spectrograms(audioFile)
            self.audio_queue.task_done()
            self.result_queue.put([mel,lin])


class spectrogramsLoader:
    def __init__(self,audioFiles,num_threads):
        self.audioFiles = audioFiles
        self.audioFilesQueue = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        # Start consumers
        if num_threads > multiprocessing.cpu_count():
            num_threads = multiprocessing.cpu_count()
        self.num_workers = num_threads if num_threads else multiprocessing.cpu_count()
        print('Creating {} consumers'.format(self.num_workers))

    def start(self):
        print("Data loading started")

        for audioFile in (self.audioFiles):
            self.audioFilesQueue.put(audioFile)

        for _ in range(self.num_workers):
            self.audioFilesQueue.put(None)
        self.workers = [
            Worker(self.audioFilesQueue, self.results)
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()
        # Wait for all of the tasks to finish
        #self.audioFilesQueue.join()
    def reset(self):
        print("data loader reset")
        for w in self.workers:
            w.terminate()
            w.join()
        print("All worker terminated")
        print("Create new queues")
        self.audioFilesQueue = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        print("Start again")
        self.start()
    def get_spectrograms(self):
        try:
            return self.results.get(block=True, timeout=30)
        except Empty:
            print(e)
            print("Queue get timeout or queue empty")
    def spectraQueueSize(self):
        return self.results.qsize()
