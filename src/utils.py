import sounddevice as sd
import config
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import streamlit as st

FRAME_SIZE = 2048
HOP_LENGTH = 512
N_MELS = 128

def record():
    duration = 3
    fs = 22050
    recording = sd.rec(frames=(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write(config.RECORDING_WAV, fs, recording)
    np.save(config.RECORDING_NPY, recording)
    print('recording complete')

def play():
    fs = 22050
    recording = np.load(config.RECORDING_NPY)
    sd.play(recording, fs)


def create_tensor():
    device = torch.device('cpu')
    audio, sr = librosa.load(config.RECORDING_WAV, offset=0.3, duration=1.5)
    melspectogram = librosa.feature.melspectrogram(audio, sr, 
                                                   n_fft=FRAME_SIZE, 
                                                   hop_length=HOP_LENGTH,
                                                   n_mels=N_MELS)
    melspectogram = librosa.power_to_db(melspectogram)
    mel_tensor = torch.from_numpy(melspectogram).type(torch.FloatTensor)
    mel_tensor = mel_tensor.view(1,1,128,65)
    mel_tensor = mel_tensor.to(device)
    return mel_tensor

def display_spectogram():
    audio, sr = librosa.load(config.RECORDING_WAV, offset=0.3, duration=1.5)
    melspectogram = librosa.feature.melspectrogram(audio, sr, 
                                                   n_fft=FRAME_SIZE, 
                                                   hop_length=HOP_LENGTH,
                                                   n_mels=N_MELS)
    melspectogram = librosa.power_to_db(melspectogram)

    plt.figure(figsize=(15, 8))
    librosa.display.specshow(melspectogram,
                            sr=sr,
                            hop_length=HOP_LENGTH,
                            x_axis='time',
                            y_axis='log')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format='%+2.0f')
    st.pyplot(plt)