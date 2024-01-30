#!/usr/bin/env python3'''
'''function that evaluates the output of a network'''
import tensorflow as tf

def evaluate(X,Y,save_path):
    '''args: x: input data
            Y: one-hot labels for x
        returns: networks pred, accuracy, loss'''
    with tf.session() as sess:
        ''''import the metagraph'''
        saver = tf.train.import_meta_graph(save_path + '/model.ckpt.meta')
        '''restore saved variables'''
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        # access the tensors from the collection
        