import matplotlib.pyplot as plt
import librosa as dsp
import librosa.display as dsp_plot
import numpy as np
import math
from params import Hyperparams as hp

def load_wave(audiofilepath):
    return dsp.load(audiofilepath,sr=hp.sr)

def do_preemphasis(wave, pre_emphasis=0.97):
    return np.append(wave[0], wave[1:] - pre_emphasis * wave[:-1])

def do_spectrogram(y=None, sr=16000,win_length=0.05,hop_length=0.0125, n_fft=2048,n_mels=80):
    #lengths: seconds -> samples
    hop_length = int(hop_length*sr);
    win_length = int(win_length*sr);

    #do stft first cause need to pass win_length arg
    stft = dsp.stft(y=y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    #####################
    #keithito version
    mels=dsp.filters.mel(sr, n_fft, n_mels=n_mels)
    s_mel = np.dot(mels,np.abs(stft))
    #read 20*log.. as 10*2 because (abs(signal))^2 => 2*10*log...
    S_mel = 20 * np.log10(np.maximum(1e-8,s_mel))
    S_lin =20 * np.log10(np.maximum(1e-8,np.abs(stft)))
    if hp.use_ref_db:
        print("Using reference db..")
        S_mel=S_mel - hp.ref_level_db
        S_lin=S_lin - hp.ref_level_db

    #normalize between 0-1 (https://github.com/keithito/tacotron/issues/38)
    if hp.normalize:
        print("Normalizing..")
        S_mel=np.clip((S_mel - hp.min_level_db) / -hp.min_level_db, 0, 1)
        S_lin=np.clip((S_lin - hp.min_level_db) / -hp.min_level_db, 0, 1)

    return S_lin,S_mel
    #print(S[:200])

def spect2wav(S,denormalize=True, use_ref_db=True,win_length=0.05,hop_length=0.0125, n_fft=2048):
    '''Converts spectrogram to waveform using librosa'''

    # Convert back to linear
    if denormalize:
        print("Denormalizing..")
        S = (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db
    print(S.shape)
    if use_ref_db:
        print("Adding ref db..")
        S=S + hp.ref_level_db

    #db -> amp
    S= np.power(10.0, S * 0.05)
    #amplitude raises of 1.5 as keithito (paper put 1.2)
    S = S**(1.5) #(1.2)

    #lengths: seconds -> samples
    hop_length = int(hop_length*sr);
    win_length = int(win_length*sr);

    #Based on https://github.com/librosa/librosa/issues/434
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = dsp.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    for i in range(hp.griffin_lim_iters):
        ft = dsp.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        angles = np.exp(1j * np.angle(ft))
        y = dsp.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    return y
