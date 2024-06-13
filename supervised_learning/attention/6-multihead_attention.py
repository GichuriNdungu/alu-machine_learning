#!/usr/bin/env python3

''''calculates the scaled dot product attention'''

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    '''class that implements multihead attention for an RNN'''

    def __init__(self, dm, h):
        '''params:
                dm: int, reps dimensionality of the model
                h:: reps number of heads
                dm is divisible by h
            rtype:
                sets public instance attributes
        '''
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        # setting the depth of each attention head ???
        self.depth = dm // h
        # query matrix
        self.Wq = tf.keras.layers.Dense(units=self.dm)
        # key matrix
        self.Wk = tf.keras.layers.Dense(units=self.dm)
        # value matrix
        self.Wv = tf.keras.layers.Dense(units=self.dm)
        # attention output
        self.linear = tf.keras.layers.Dense(units=self.dm)

    def split_heads(self, x, batch_size):
        '''split the last dimension of x to facilitate parallel computation'''
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
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
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        attention, attention_weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.dm))
        outputs = self.linear(concat_attention)

        return outputs, attention_weights
