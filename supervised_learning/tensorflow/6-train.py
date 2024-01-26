#!/usr/bin/env python3'''
'''function that builds, trains, and saves a neural network classifier'''

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop
import tensorflow as tf

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''args: X_train: training input data
    Y_train: training labels
    X_valid: validation input data
    Y_valid: validation labels
    layer_sizes: list containing the number of nodes in each layer of the network
    activations: list containing the activation functions for each layer of the network
    alpha: learning rate
    iterations: number of iterations to train over
    save_path: designates where to save the model
    returns: the path where the model was saved'''
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    # Calculate accuracy
    accuracy = calculate_accuracy(y, y_pred)
    # Calculate loss
    loss = calculate_loss(y, y_pred)
    # Create train operation
    train_op = create_train_op(loss, alpha)
    # Initialize variables
    init = tf.global_variables_initializer()
    # Create saver
    saver = tf.train.Saver()
    # Create session
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            # Calculate cost and accuracy for training data
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            # Calculate cost and accuracy for validation data
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            # Print costs and accuracies
            if i % 100 == 0 or iterations == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))