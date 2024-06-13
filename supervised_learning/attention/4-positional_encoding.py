#!/usr/bin/env python3

'''calculates the positional encodings for each word in a sequence'''

import numpy as np


def get_angle(pos, i, dm):
    '''calculates the angles for positional encoding formulas
    PE(pos, 2i) = sin(pos / 10000^(2i/dm))
    PE(pos, 2i +1) = cos(pos / 10000^(2i/dm))'''

    angle_rates = 1 / (10000 ** (i / dm))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    '''calculates the positional encoding for a transformer
    params:
        max_seq_len: maximum sequence length
        dm: model depth
    return:
        np array with shape(max_seq_len, dm)
            array is positional encoding vectors'''
    PE = np.zeros([max_seq_len, dm])
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            PE[pos, i] = np.sin(get_angle(pos, i, dm))
            PE[pos, i+1] = np.cos(get_angle(pos, i, dm))
    return PE
import tensorflow as tf

def positional_encoding(input_tensor, max_len=512):
    """
    Adds positional encoding to the input tensor.

    Args:
    input_tensor: Input tensor of shape (batch_size, input_seq_len, dm).
    max_len: Maximum sequence length for positional encoding (default: 512).

    Returns:
    Tensor: Input tensor with positional encoding added.
    """

    batch_size, input_seq_len, dm = input_tensor.shape

    # Initialize positional encoding matrix
    pos_encoding = tf.zeros((max_len, dm))

    # Calculate positional encodings
    positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    div_terms = tf.pow(10000.0, tf.range(0, dm, 2, dtype=tf.float32) / dm)

    pos_encoding = tf.concat([tf.sin(positions / div_terms), tf.cos(positions / div_terms)], axis=-1)

    # Ensure pos_encoding has the correct shape (max_len, dm)
    pos_encoding = pos_encoding[:max_len, :]

    # Add batch dimension and slice to input_seq_len
    pos_encoding = pos_encoding[tf.newaxis, :input_seq_len, :]

    # Add positional encodings to input embeddings
    output_tensor = input_tensor + pos_encoding

    return output_tensor

# Example usage:
input_tensor = tf.random.normal((32, 100, 256))  # Example input tensor
output_tensor = positional_encoding(input_tensor, max_len=128)  # Add positional encoding
 

