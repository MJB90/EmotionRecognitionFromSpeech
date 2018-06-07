import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram


def load_sound_files(file_paths):
    raw_sounds=[]
    for fp in file_paths:
        X,sr=librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,100), dpi = 100)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(4,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=1,fontsize=18)
    plt.show()


def plot_specgram(sound_names,raw_sounds):
    i=1
    fig=plt.figure(figsize=(25,100),dpi=100)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(4,1,i)
        specgram(np.array(f),Fs=22050)
        plt.title(n.title())
        i+=1
    plt.suptitle("Figure 1: Spectogram", x=0.5, y=1, fontsize=18)
    plt.show()


sound_file_paths = ["03a02Nc.wav","03a02Ta.wav","03a02Wb.wav"]

sound_names = ["fast","normal","angry"]

raw_sounds=load_sound_files(sound_file_paths)
#plot_specgram(sound_names,raw_sounds)
plot_waves(sound_names,raw_sounds)
# data,sr=librosa.load(sound_file_paths[2])
# fig = plt.figure(figsize=(25,60), dpi = 900)
# librosa.display.waveplot(data,sr)
# plt.show()


