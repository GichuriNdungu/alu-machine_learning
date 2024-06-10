#!/usr/bin/env python3
'''implements the decoder for the transformer model'''
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    '''class that defines the encoder part of a seq2seq model'''

    def __init__(self, vocab, embedding, units, batch):
        '''initializer of class RNNDecoder'''
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        '''params:
                x: tensor containing the previous word in target sequence
                s_prev: hidden state of the previous decoder
                hidden_states: tensor with the outputs of the encoder
            return:
                y: output vector as one hot vector in target vocab
                s: new decoder hidden state'''
        units = s_prev.get_shape().as_list()[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
