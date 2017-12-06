#import matplotlib.pyplot as plt
import librosa as dsp
#import librosa.display as dsp_plot
import numpy as np
import math
from params import Hyperparams as hp

def load_wave(audiofilepath):
    return dsp.load(audiofilepath,sr=hp.sr)
def save_wave(file_path,waveform,sr):
	dsp.output.write_wav(file_path, waveform, sr,norm=True)
def spectrogram(magn):
    #D = dsp.stft(y=y, n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length)
    S = _amp_to_db(magn) - hp.ref_level_db
    return _normalize(S)

def melspectrogram(magn):
    #D = dsp.stft(y=y, n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length)
    S = _amp_to_db(_linear_to_mel(magn))
    return _normalize(S)

def do_spectrograms(y):
    magn = np.abs(dsp.stft(y=y, n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length))
    return spectrogram(magn), melspectrogram(magn)

def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hp.ref_level_db)  # Convert back to linear
  return _griffin_lim(S ** hp.power)          # Reconstruct phase


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = dsp.istft(S_complex * angles, hop_length=hp.hop_length, win_length=hp.win_length)
  for i in range(hp.griffin_lim_iters):
    angles = np.exp(1j * np.angle(dsp.stft(y=y, n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length)))
    y = dsp.istft(S_complex * angles, hop_length=hp.hop_length, win_length=hp.win_length)
  return y


def do_preemphasis(wave, pre_emphasis=0.97):
    return np.append(wave[0], wave[1:] - pre_emphasis * wave[:-1])

def _normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def _denormalize(S):
    print("Denormalizing...")
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):

    print("Convert from dB to amp..",x.shape)
    return np.power(10.0, x * 0.05)

_mel_basis = None
def get_mel_basis():
    return dsp.filters.mel(hp.sr, hp.n_fft, n_mels=hp.n_mels)

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = get_mel_basis()
    return np.dot(_mel_basis, spectrogram)
