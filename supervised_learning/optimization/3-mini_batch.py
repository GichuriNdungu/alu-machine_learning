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
        steps = round(len(X_train)/ batch_size)
        m = X_train.shape[0]

        for epoch in range(epochs+1):
            
            feed_train = {x:X_train, y:Y_train}
            Valid_train = {x:X_valid, y:Y_valid}

            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)

            valid_cost = sess.run(loss, feed_dict=Valid_train)
            valid_accuracy = sess.run(accuracy, feed_dict=Valid_train)

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            if epoch != epochs:
                start = 0
                end = batch_size

                X_trains, Y_trains = shuffle_data(X_train, Y_train)
            
                for step in range(1, steps+2):
                    x_batch = X_trains[start: end]
                    y_batch = Y_trains[start: end]
                    temp_feed = {x:x_batch, y:y_batch}
                    # back_propagation
                    train = sess.run(train_op, feed_dict={x:x_batch, y:y_batch})
                    # back propagation is applied to each of the steps 
                    # therefore, check whether the steps are divisible by 100
                    if step %100 == 0:
                        t_cost = sess.run(loss, feed_dict=temp_feed)
                        step_accuracy = sess.run(accuracy, feed_dict =temp_feed)
                        print(f'\t\tstep: {step} \n'
                            f'\t\tCost: {t_cost} \n'
                            f'\t\tAccuracy: {step_accuracy}')
                    start = start + batch_size
                    # handle instances when the last batch is not equal to batch size
                    if (m-start) < batch_size:
                        end = end + (m-start)
                    else:
                        end = end+batch_size
        save_path = saver.save(sess, save_path)
    return save_path