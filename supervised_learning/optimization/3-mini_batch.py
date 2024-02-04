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
        for epoch in range(epochs):
            
            feed_train = {x:X_train, y:Y_train}
            Valid_train = {x:X_valid, y:Y_valid}

            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)

            valid_cost = sess.run(loss, feed_dict=valid_cost)
            valid_accuracy = sess.run(accuracy, feed_dict=Valid_train)
            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            X_trains, Y_trains = shuffle_data(X_train, Y_train)
            step_counter = 0
            
            for batch in range(0, len(X_trains),batch_size):
                x_batch = X_trains[batch: batch+batch_size]
                y_batch = Y_trains[batch: batch+batch_size]
                # back_propagation
                cost, _ = sess.run([loss, train_op], feed_dict={x:x_batch, y:y_batch})
                # count the steps of back_propagation
                step_counter += 1
                if (step_counter+1)%100 and step_counter != 0:
                    step_accuracy = sess.run(accuracy, feed_dict ={x:x_batch, y:y_batch})
                    print(f'\t\tstep: {step_counter} \n'
                          f'\t\tCost: {cost} \n'
                          f'\t\tAccuracy: {step_accuracy}')     
        save_path = saver.save(sess, save_path)
    return save_path