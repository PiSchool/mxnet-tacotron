# coding: utf-8
import csv

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
# Returns: one-hot-encoded-text, linear spectrum, mel spectrum
# Shapes: (data_length, ?, ?) , (data_length, (n_fft/2)+1, ceil(max_audio_length/hop_size)), (data_length, n_mels, ceil(max_audio_length/hop_size))
def generate_text_spectra(texts_list, sound_labels):

    assert len(sound_labels) == len(texts_list)

    print("Generating spectrograms")
    print("Sample length for windowing:",hp.win_length)
    print("Sample length for hop:",hp.hop_length,"\n")
    #tuples of wav and sr of that wav. wav is a 1D floats vector
    wavs_srs = [audio_process.load_wave(sound_clip) for sound_clip in sound_labels]
    longest_wav_sr = (max(wavs_srs, key= lambda wav: len(wav[0])))
    #save the longest audio file length
    max_samples_length=(len(longest_wav_sr[0]))
    print("max audio sample length:",max_samples_length)

    #prepare the data structure for save all the spectra
    spectra_lin = []#mx.ndarray.zeros((len(sound_labels),math.ceil(max_samples_length/hp.hop_length),1+(hp.n_fft//2)))
    spectra_mel = []#mx.ndarray.zeros((len(sound_labels),math.ceil(max_samples_length/hp.hop_length),hp.n_mels))
    mel_basis = audio_process.get_mel_basis()
    print("Padding audio and compute mel and lin spectra..")
    for indx,wav_sr in enumerate(wavs_srs):
        wav = wav_sr[0]
        wav_length = len(wav)
#         print("wav l",w_length)
        diff = max_samples_length-wav_length
#         print("num of zeros to add",diff)
        padded = np.append(wav,np.zeros(diff))
        # get the spectrum from the padded sound
        spectrum_lin, spectrum_mel=audio_process.do_spectrograms(y=padded)
#         print(padded_spectrum_lin.shape)
        # save into the ndarray
        spectra_lin.append(np.transpose(spectrum_lin))#[indx,:,:]=np.transpose(spectrum_lin[:,:])
        spectra_mel.append(np.transpose(spectrum_mel))#[indx,:,:]=np.transpose(spectrum_mel[:,:])



    texts_one_hot=None
    if hp.do_text_processing:
        print("Processing text..")
        vocabulary = generate_vocabulary(texts_list)
        vocab_size=len(vocabulary)
        char2index,index2char = generate_chars2numbers_mappings(vocabulary)

        print("Converting text to integers..")
        texts_numerical = text2numbers(texts_list,char2index)
        # simulate a different sequence length
    #   /D E L E T E M E/
        texts_numerical[4]=np.concatenate((texts_numerical[4],[8,9]))
    #   /D E L E T E M E/

        longest_sequence = (max(texts_numerical, key= lambda seq: len(seq)))
        longest_sequence_len=len(longest_sequence)
        print("Pad sequences to",longest_sequence_len,"..")
        # helper function for the lambda expression
        def _padseq(seq,max_len):
            diff=max_len-len(seq)
            if diff>0:
                # SHITTY USELESS MXNET API. CANNOT CONCAT A NON-EMPTY WITH EMPTY ARRAY.
                # EDIT: use numpy now. Still using this condition for safety
                pad = np.zeros(diff)-1
                seq=np.append(seq,[pad])
            return seq

        padded_sequences = mx.nd.array(
            list(
                map(
                    lambda seq: _padseq(seq,longest_sequence_len), texts_numerical
                )
            )
        )

        texts_one_hot=mx.ndarray.one_hot(padded_sequences,vocab_size)

    return texts_one_hot, mx.nd.array(spectra_lin), mx.nd.array(spectra_mel)


# In[3]:



#class NDArrayIter_NTC(mx.io.NDArrayIter):

#    @property
#    def provide_data(self):
#        """The name and shape of data provided by this iterator."""
#        return [
#            mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype, layout="NTC")
#            for k, v in self.data
#        ]



# In[4]:


def get_iterators():
    texts_list, sound_files_list = open_data(hp.csv_file)
    size=len(sound_files_list)

    texts_one_hot, spectra_lin, spectra_mel = generate_text_spectra(texts_list, sound_files_list)

    # get 10% of dataset as eval data
    eval_indxs = (np.random.randint(0, high=size, size=size//10))
    #eval_indxs=[32 10 28 19 29]
    # remaining indexes for the train
    train_indxs = np.setdiff1d(np.arange(size),eval_indxs)

    print("I will take those for eval:",eval_indxs)
    print("..and the remaining for train:",train_indxs,"\n")

    #take from the array (1st arg) the indexes of the first dimension specified by the 2nd arg
    #train_txt take the one_hot matrices

    if hp.do_text_processing:
        train_txt_data = mx.ndarray.take(texts_one_hot,mx.nd.array(train_indxs))
        eval_txt_data = mx.ndarray.take(texts_one_hot,mx.nd.array(eval_indxs))

    train_data = mx.ndarray.take(spectra_mel,mx.nd.array(train_indxs))
    train_label = mx.ndarray.take(spectra_lin,mx.nd.array(train_indxs))

    eval_data = mx.ndarray.take(spectra_mel,mx.nd.array(eval_indxs))
    eval_label = mx.ndarray.take(spectra_lin,mx.nd.array(eval_indxs))

    print("train data shape:",train_data.shape,"train label shape:",train_label.shape)
    print("eval data shape:", eval_data.shape,"eval label shape:",eval_label.shape,"\n")


    try:
        print("Populating traindata iterator")
        traindata_iterator = mx.io.NDArrayIter(data={'mel_spectrogram':train_data},
                                label={'linear_spectrogram':train_label},
                                batch_size=hp.batch_size,
                                shuffle=True)
        print("Populating evaldata iterator")
        evaldata_iterator = mx.io.NDArrayIter(data={'mel_spectrogram':eval_data},
                                label={'linear_spectrogram':eval_label},
                                batch_size=hp.batch_size)
    except Exception as e:
        print(e)
        traceback.print_exc()

#     for batch in traindata_iterator:
#         print(batch.data[0].asnumpy())
#         print(batch.data[0].shape)

    return traindata_iterator,evaldata_iterator, train_data.shape[1],eval_data,eval_label

