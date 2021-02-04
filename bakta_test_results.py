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

from bangla_tts import spectrogram2wav, griffin_lim, invert_spectrogram, upsample2, generate, generate_long


import json

bang_f = json.load(open("test_dataset/bakta_bang.json", encoding='utf-8'))

bangeng_f = json.load(open("test_dataset/bakta_bang_eng.json", encoding='utf-8'))

cnt = 1
for sen in bang_f:
    # print(sen)
    generate_long(sen["Sentence"], f"test_dataset/results/bangla_tts_bang{cnt}.wav")
    cnt += 1
    time.sleep(1)

cnt = 1
for sen in bangeng_f:
    # print(sen)
    generate_long(sen["Sentence"], f"test_dataset/results/bangla_tts_bangeng{cnt}.wav")
    cnt += 1