# python 3.6
# github/zabir-nabil

import os

from config import *
import numpy as np
import tensorflow as tf
from graph import Graph

from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
import time
import random

import numpy as np
import librosa
import os, copy

from scipy import signal

import requests
import shutil
import wget

import warnings
warnings.filterwarnings("ignore")
import sys, os


# linear spectogram to wav (temporal speech)

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


# for doubling the temporal points, for generating longer signals in low resolution

def upsample2(m2):
    x = []
    x.append(m2[0])
    cnt = 0
    for ix in m2:
        if cnt == len(m2)-1:
            break
        x.append((x[-1] + m2[cnt+1])/2)
        x.append(m2[cnt+1])
        cnt += 1
    x.append(x[-1])
    x = np.array(x)
    return x

# speech synthesis class

def generate(text_arr=[""], save_path = None):
    '''
    function: generate(text_arr = [""], save_path = None)
    arguments: 
    text_arr (array) : an array of strings
    save_path (string, optional) : location where generated wav files will be stored if save_path is not None, if the path is not valid, the wav files will be saved in current directory
    returns:
    if save_path is None, instead of saving an array of tuples containing geenrated speech signals and the sampling rate will be returned
    if save_path is not None, then a list containing the file paths (relative) will be returned
    '''

    # the weights couldn't be stored directly in github
    if not os.path.exists("model1/model_gs_301k.data-00000-of-00001"):
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print("No weights found for first model. Downloading ...")
        wget.download("https://gitlab.com/zabir-nabil/bangla_tts_weights/raw/master/model_gs_301k.data-00000-of-00001")
        shutil.move("model_gs_301k.data-00000-of-00001", "model1/model_gs_301k.data-00000-of-00001")

    if not os.path.exists("model2/model_gs_300k.data-00000-of-00001"):
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print("No weights found for second model. Downloading ...")
        wget.download("https://gitlab.com/zabir-nabil/bangla_tts_weights/raw/master/model_gs_300k.data-00000-of-00001")
        shutil.move("model_gs_300k.data-00000-of-00001", "model2/model_gs_300k.data-00000-of-00001")

    # Load data
    L = load_data(text_arr)



    # Load graph
    g = Graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        # check for the weights


        saver1.restore(sess, tf.train.latest_checkpoint("model1"))
        print("Model 1 loaded!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint("model2"))
        print("Model 2 loaded!")



        t1 = time.time()

        ## mel generation
        Y = np.zeros((len(L), max_T, n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude spectrum
        Z = sess.run(g.Z, {g.Y: Y})

      
        generated_wav = [] # a tuple, wav numpy array and sampling rate
        file_paths = []
        for i, mag in enumerate(Z):
            #mag = upsample2(mag)
            wav = spectrogram2wav(mag) # griffin-lim speech generation

            pp = random.randint(1,1000000) # generate a random secondary ID for the audio (for avoiding caches)
            pp = str(i) + '_' + str(pp)

            if save_path is not None:
                if os.path.exists(save_path):
                    write(save_path + "/{}.wav".format(pp), sr, wav)
                    file_paths.append(save_path + "/{}.wav".format(pp))
                else:
                    write("{}.wav".format(pp), sr, wav) # save to pwd
                    file_paths.append("{}.wav".format(pp))
            
            if save_path is None:
                generated_wav.append((wav, sr))

        t_needed = time.time() - t1

        print(f'Total time taken {t_needed} secs.')

        if save_path is None:
            return generated_wav # send as an array (wav, sampling rate)
        else:
            return file_paths


def generate_long(text="", save_path = "out.wav", numeric_translation = True): # slower, but can translate numeric details and longer sentences
    import num_parser
    """
    params: text :: a str (long)
            numeric_translation :: phonetic translation will be performed before speech generation [slightly slower]

            ** will be saved as out.wav **
    """
    # the weights couldn't be stored directly in github
    if not os.path.exists("model1/model_gs_301k.data-00000-of-00001"):
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print("No weights found for first model. Downloading ...")
        wget.download("https://gitlab.com/zabir-nabil/bangla_tts_weights/raw/master/model_gs_301k.data-00000-of-00001")
        shutil.move("model_gs_301k.data-00000-of-00001", "model1/model_gs_301k.data-00000-of-00001")

    if not os.path.exists("model2/model_gs_300k.data-00000-of-00001"):
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print("No weights found for second model. Downloading ...")
        wget.download("https://gitlab.com/zabir-nabil/bangla_tts_weights/raw/master/model_gs_300k.data-00000-of-00001")
        shutil.move("model_gs_300k.data-00000-of-00001", "model2/model_gs_300k.data-00000-of-00001")

    text_arr = num_parser.process(text)
    print(text_arr)

    # Load data
    L = load_data(text_arr)



    # Load graph
    tf.reset_default_graph()
    g = Graph()

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        # check for the weights


        saver1.restore(sess, tf.train.latest_checkpoint("model1"))
        print("Model 1 loaded!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint("model2"))
        print("Model 2 loaded!")



        t1 = time.time()

        ## mel generation
        Y = np.zeros((len(L), max_T, n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude spectrum
        Z = sess.run(g.Z, {g.Y: Y})

      
        generated_wav = np.array([]) # a tuple, wav numpy array and sampling rate

        for i, mag in enumerate(Z):
            #mag = upsample2(mag)
            wav = spectrogram2wav(mag) # griffin-lim speech generation

            generated_wav = np.append(generated_wav, wav)

        t_needed = time.time() - t1

        print(f'Total time taken {t_needed} secs.')

        write(save_path, sr, generated_wav)


if __name__ == '__main__':
    # generate(["আমার সোনার বাংলা আমি তোমাকে ভালোবাসি", "আমার নাম জাবির আল নাজি নাবিল", "I am still not a great speaker", "This is just a test"], 'static')
    # generate_long("বাংলাদেশে গত ২৪ ঘণ্টায় ৩০৬ জন কোভিড-১৯ আক্রান্ত হয়েছেন। এই সময়ের মধ্যে মৃত্যু হয়েছে ৯ জনের। এ নিয়ে দেশটিতে মোট আক্রান্ত হলেন ২১৪৪। আর করোনা ভাইরাসে আক্রান্ত হয়ে মৃত্যু হয়েছে ৮৪ জনের। নতুন করে ৮ জনের পরীক্ষা করার পর করোনা ভাইরাসের উপস্থিতি পাওয়া যায়নি। এনিয়ে মোট ৬৬ জন সুস্থ হলেন।")
    # generate_long("আমার ফোন নাম্বার ০১৭১৩৩৫৩৪৩, তবে আমাকে সকাল ১০ টার আগে পাবেন না")
    generate_long("১৯৯৭ সালের ২১ জানুয়ারী তে আমার জন্ম হয়")
    # sentence credit: BBC - Bangla


