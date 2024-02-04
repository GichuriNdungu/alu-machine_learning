#!/usr/bin/env python3
'''function that trains a
 model using mini-batch gradient descent'''
shuffle_data = __import__('2-shuffle_data').shuffle_data
import tensorflow as tf
def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                      batch_size=32, epochs=5,
                        load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        valid_cost = tf.get_collection('valid_cost')[0]
        valid_accuracy = tf.get_collection('valid_accuracy')[0]
        initial_loss = sess.run(loss, feed_dict={x:X_train, y:Y_train})
        print(f'After {epoch} epochs:\n'
            f'\tTraining Cost: {initial_loss}\n'
            f'\tTraining Accuracy: {accuracy}\n'
            f'\tValidation Cost: {valid_cost}\n'
            f'\tValidation Accuracy: {valid_accuracy}')
        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            step_counter = 0
            for batch in range(0, len(X_train),batch_size):
                x_batch = X_train[batch: batch+batch_size]
                y_batch = Y_train[batch: batch+batch_size]
                # forward propagation
                predictions = sess.run(y, feed_dict={x: x_batch})
                # back_propagation
                cost, _ = sess.run([loss, train_op], feed_dict={x:x_batch, y:y_batch})
                # count the steps of back_propagation
                step_counter += 1
                if step_counter%100:
                    step_accuracy = sess.run(accuracy, feed_dict ={x:x_batch, y:y_batch})
                    print(f'\t\tCost: {cost} \n'
                          f'\t\tCost: {cost} \n'
                          f'\t\tAccuracy: {step_accuracy}')
        print(f'After {epoch} epochs:\n'
            f'\tTraining Cost: {initial_loss}\n'
            f'\tTraining Accuracy: {accuracy}\n'
            f'\tValidation Cost: {valid_cost}\n'
            f'\tValidation Accuracy: {valid_accuracy}')       
        saver.save(sess, save_path)
        return save_path