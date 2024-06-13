#!/usr/bin/env python3

''''creates an encoder block for a transformer'''

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    '''class that implements a full encoder block for a transformer'''

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        '''class constructor
        params:
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in fully connected layer
            drop_rate: dropout rate
        rtpte:
            sets public instance attributes'''
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.dropout = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def positional_encoding(input_tensor, max_len=512):
        '''performs positonal encoding for the encoding block
        params:
            input_tensor: input tensor
            max_len: maximum sequence length
        return:
            Tensor: input tensor with positiona encoding added'''
        bath_size, input_seq_len, dm = tf.shape(input_tensor)

        # initialize positional encoding matrix

        pos_encoding = tf.zeros((max_len, dm))

        # calculate positional encodings
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_terms = tf.pow(10000.0, tf.range(0, dm, 2, dtype=tf.float32) // dm)
        pos_encoding = tf.concat(
            [tf.sin(positions / div_terms),
             tf.cos(positions/div_terms)], axis=-1)
        # ensure pos_encoding has the correct shape(max_len, dm)
        pos_encoding = pos_encoding[:max_len, :]
        pos_encoding = pos_encoding[tf.newaxis, :input_seq_len, :]
        output_tensor = input_tensor + pos_encoding

        return output_tensor

    def call(self, x, training, mask=None):
        '''call method for the encoder block'''
        # perform positional encoding for the input
        # x = self.positional_encoding(x)
        # now using the pre-defined multihead layer, implement mha
        mha_outputs, _ = self.mha(x, x, x, mask)
        mha_outputs = self.dropout1(mha_outputs, training=training)
        normalized_output1 = self.layernorm1(x + mha_outputs)

        dense_output = self.dense_hidden(normalized_output1)
        feed_forward = self.dense_output(dense_output)
        feed_forward_dropout = self.dropout2(feed_forward, training=training)
        final_output = self.layernorm2(
            normalized_output1 + feed_forward_dropout)

        return final_output
