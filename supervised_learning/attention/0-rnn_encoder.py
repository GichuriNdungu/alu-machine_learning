#!/usr/bin/env python3

'''implements the encoder for the seq2seq model'''

import tensorflow as tf

class RNNEncoder(tf.keras.layers.Layer):
    '''class that defines the encoder part of a seq2seq model'''
    def __init__(self, vocab, embedding, units, batch):
        '''initializer of class RNNEncoder'''
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab, output_dim = embedding)
        self.gru = tf.keras.layers.GRU(units = units, kernel_initializer = 'glorot_uniform', return_sequences = True, return_state = True)
    def initialize_hidden_state(self):
        '''initializers the hidden states for the RNN cell to a tensor of zeros'''
        return tf.zeros(shape = (self.batch, self.units))
    def call(self, x, initial):
        '''params:
                x:tensor containing the input to the encoding layer as word indices to the vocab
                initial: tensor containing the initial hidden states
            return:
                outputs: tensor containing outputs from the encoder
                hidden: tensor containing the last hidden state of the hidden layer'''
        x = self.embedding(x)
        if initial is None:
            initial = self.initialize_hidden_state()
        outputs, hidden = self.gru(x, initial_state = initial)
        return outputs, hidden

