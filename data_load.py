# python 3.6
# github/zabir-nabil

from config import *
import numpy as np
import tensorflow as tf
import codecs
import re
import os
import unicodedata

# text to numeric tensor, normalization

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in text)

    text = text.lower()
    text = re.sub("[^{}]".format(vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(text_arr=[""]):
    # Load vocabulary
    char2idx, _ = load_vocab()
    lines = []
    tid = 1
    for text in text_arr:
        lines.append(str(tid) + ". " + text + '   ') # empty space == silence
        tid += 1
    # print(lines)

    sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
    texts = np.zeros((len(sents), max_N), np.int32)
    for i, sent in enumerate(sents):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts

