#!/usr/bin/env python3

''''creates an Decoder block for a transformer'''

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
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
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.dropout = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        '''call method for the Decoder block class
            params:
                x: input to the encoder block
                encoder_output:  output from the encoder block
                training: Boolean to determine whether model is training
                look_ahead_mask: mask for first multihead att layer
                padding_mask: mask for the second multihead att layer
            rtype:
                Tensor: block's output'''
        # masked multihead attention
        att1, _ = self.mha1(x, x, x, look_ahead_mask)
        att1_drop = self.dropout1(att1, training=training)
        out_1 = self.layernorm1(x + att1_drop)

        # multihead attention
        att2, _ = self.mha2(out_1, encoder_output,
                            encoder_output, padding_mask)
        att2_drop = self.dropout2(att2, training=training)
        out_2 = self.layernorm2(out_1 + att2_drop)

        # feed forward neural net

        dense_output = self.dense_hidden(out_2)
        feed_forward = self.dense_output(dense_output)
        feed_forward_dropout = self.dropout3(feed_forward, training=training)
        final_output = self.layernorm3(out_2 + feed_forward_dropout)

        return final_output
