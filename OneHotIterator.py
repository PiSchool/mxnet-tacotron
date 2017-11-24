import mxnet as mx

# THIS IS A VERY SIMPLE ITERATOR THAT, TAKEN A LIST OF SENTENCES ENCODED AS INTEGERS, OUTPUTS THE BATCHES IN ONE HOT FORM (TO OVERCOME MEMORY ALLOCATION ISSUES)

class OneHotIterator(mx.io.DataIter):
    def __init__(self,
                 data, label,
                 data_names, max_len_data, vocab_size_data,
                 label_names, max_len_label, vocab_size_label,
                 batch_size=10):
        self._provide_data = [
            mx.io.DataDesc(
                name=data_name,
                shape=(batch_size, max_len_data, vocab_size_data),
                layout='NTC') for data_name in data_names
        ]
        self._provide_label = [
            mx.io.DataDesc(
                name=label_names,
                shape=(batch_size, max_len_label, vocab_size_label),
                layout='NTC')
        ]
        self.num_batches = len(data)//batch_size
        self.batch_size = batch_size
        self.cur_data_pointer = 0
        self.cur_batch = 0
        self.vocab_size_data = vocab_size_data
        self.vocab_size_label = vocab_size_label
        self.data = data
        self.label = label

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0
        self.cur_data_pointer = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches-1:
            self.cur_batch += 1

            data_batch = []
            label_batch = []

            for i in range(self.batch_size):
                data_batch.append(self.data[self.cur_data_pointer])
                label_batch.append(self.label[self.cur_data_pointer])
                self.cur_data_pointer+=1

            label = [mx.nd.one_hot(mx.nd.array(data_batch), self.vocab_size_label)]
            data = [mx.nd.one_hot(mx.nd.array(data_batch), self.vocab_size_data)] + \
                [mx.nd.one_hot(mx.nd.array(label_batch), self.vocab_size_label)]

            return mx.io.DataBatch(
                data,
                label,
		pad=0,
                provide_data=self._provide_data,
                provide_label=self._provide_label
            )
        else:
            raise StopIteration

import word_utils
