#!/usr/bin/env python3

'''implements the attention layer through self attention'''

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    '''class that implements self attention for the model'''

    def __init__(self, units):
        '''params:
                units: integer, number of hidden units
                  in the alignment model'''
        super(SelfAttention, self).__init__()
        # w dense applied to the previous decoder's hidden state
        self.W = tf.keras.layers.Dense(units=units)
        # u dense applied to encoder's hidden state
        self.U = tf.keras.layers.Dense(units=units)
        # v dense applied to the tanh of the sum of w and u
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        '''params:
            s_prev: hidden state of the previous decoder
            hidden_states: outputs from the encoder
        return:
            context: context vector for the decoder
            weights:tensor with the attention weights'''
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W+U))
        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
