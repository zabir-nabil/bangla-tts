# python 3.6
# github/zabir-nabil

from tqdm import tqdm

from data_load import load_vocab
from config import *
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
import sys

# loading the networks and variables

class Graph:
    def __init__(self):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()



        self.L = tf.placeholder(tf.int32, shape=(None, None))
        self.mels = tf.placeholder(tf.float32, shape=(None, None, n_mels))
        self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))


        # network 1

        with tf.variable_scope("Text2Mel"):
            # Get S or decoder inputs. (B, T//r, n_mels)
            self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

            # Networks
            with tf.variable_scope("TextEnc"):
                self.K, self.V = TextEnc(self.L)  # (N, Tx, e)

            with tf.variable_scope("AudioEnc"):
                self.Q = AudioEnc(self.S)

            with tf.variable_scope("Attention"):
                # R: (B, T/r, 2d)
                # alignments: (B, N, T/r)
                # max_attentions: (B,)
                self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                            mononotic_attention=True,
                                                                            prev_max_attentions=self.prev_max_attentions)
            with tf.variable_scope("AudioDec"):
                self.Y_logits, self.Y = AudioDec(self.R) # (B, T/r, n_mels)

        # network 2

        # During inference, the predicted melspectrogram values are fed.
        with tf.variable_scope("SSRN"):
            self.Z_logits, self.Z = SSRN(self.Y)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)


