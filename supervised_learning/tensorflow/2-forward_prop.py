#!/usr/bin/env python3'''
''''function that does a forward propagation 
for a single layer neural network'''
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer
def forward_prop(x, layer_sizes=[], activations=[]):
    '''args:x:placeholder for the input data
       layer_sizes:list containing the number of nodes in each layer
       activations:list containing the activation functions for each layer
       returns:the prediction of the network in tensor form'''
    current_layer = create_layer(x, layer_sizes[0], activations[0])
    for size, activation in zip(layer_sizes[1:], activations[1:]):
        current_layer = create_layer(current_layer, size, activation)
    return current_layer

