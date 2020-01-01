from config import *
from modules import *
import tensorflow as tf

# network implementation is taken from https://github.com/Kyubyong/dc_tts
# paper link: https://arxiv.org/pdf/1710.08969.pdf

def TextEnc(L, training=False):
    '''
    Args:
      L: Text inputs. (B, N)

    Return:
        K: Keys. (B, N, d)
        V: Values. (B, N, d)
    '''
    i = 1
    tensor = embed(L,
                   vocab_size=len(vocab),
                   num_units=e,
                   scope="embed_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    filters=2*d,
                    size=1,
                    rate=1,
                    dropout_rate=dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=dropout_rate,
                            activation_fn=None,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    K, V = tf.split(tensor, 2, -1)
    return K, V

def AudioEnc(S, training=False):
    '''
    Args:
      S: melspectrogram. (B, T/r, n_mels)

    Returns
      Q: Queries. (B, T/r, d)
    '''
    i = 1
    tensor = conv1d(S,
                    filters=d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            padding="CAUSAL",
                            dropout_rate=dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=3,
                        padding="CAUSAL",
                        dropout_rate=dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    return tensor

def Attention(Q, K, V, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)
      mononotic_attention: A boolean. At training, it is False.
      prev_max_attentions: (B,). At training, it is set to None.

    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      alignments: (B, N, T/r)
      max_attentions: (B, T/r)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.rsqrt(tf.to_float(d))
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions, max_N)
        reverse_masks = tf.sequence_mask(max_N - attention_win_size - prev_max_attentions, max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, max_T, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)
    A = tf.nn.softmax(A) # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)

    return R, alignments, max_attentions

def AudioDec(R, training=False):
    '''
    Args:
      R: [Context Vectors; Q]. (B, T/r, 2d)

    Returns:
      Y: Melspectrogram predictions. (B, T/r, n_mels)
    '''

    i = 1
    tensor = conv1d(R,
                    filters=d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(4):
        tensor = hc(tensor,
                        size=3,
                        rate=3**j,
                        padding="CAUSAL",
                        dropout_rate=dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    for _ in range(3):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    # mel_hats
    logits = conv1d(tensor,
                    filters=n_mels,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    Y = tf.nn.sigmoid(logits) # mel_hats

    return logits, Y

def SSRN(Y, training=False):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)

    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''

    i = 1 # number of layers

    # -> (B, T/r, c)
    tensor = conv1d(Y,
                    filters=c,
                    size=1,
                    rate=1,
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(2):
        tensor = hc(tensor,
                      size=3,
                      rate=3**j,
                      dropout_rate=dropout_rate,
                      training=training,
                      scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor,
                                  scope="D_{}".format(i),
                                  dropout_rate=dropout_rate,
                                  training=training,); i += 1
        for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    # -> (B, T, 2*c)
    tensor = conv1d(tensor,
                    filters=2*c,
                    size=1,
                    rate=1,
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    # -> (B, T, 1+n_fft/2)
    tensor = conv1d(tensor,
                    filters=1+n_fft//2,
                    size=1,
                    rate=1,
                    dropout_rate=dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    logits = conv1d(tensor,
               size=1,
               rate=1,
               dropout_rate=dropout_rate,
               training=training,
               scope="C_{}".format(i))
    Z = tf.nn.sigmoid(logits)
    return logits, Z
