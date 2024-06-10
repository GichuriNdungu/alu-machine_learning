#!/usr/bin/env python3

''''calculates the scaled dot product attention'''

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    '''
    params:
        Q: tensor containing the query matrix
        V:tensor containing the Value matrix
        K:tensor containing the Key matrix
        mask: optional mask
    returns:
        output with the scaled dot product attention
        weights: attention weights tensor
        '''
    temperature = tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
    attn = tf.matmul(Q / temperature, K, transpose_b=True)
    if mask is not None:
        attn = tf.where(tf.equal(mask, 0), -1.e9, attn)
    attn = tf.nn.softmax(attn, axis=-1)
    output = tf.matmul(attn, V)
    return output, attn
