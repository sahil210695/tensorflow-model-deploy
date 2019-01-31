# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:30:41 2019

@author: sahil210695

1. Train and save model
2. Save model summary to visualize graph in tensorboard
"""

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

EPOCH = 35
LEARING_RATE = 0.01
MODEL_NAME = 'linear'
MODEL_SAVE_PATH = os.path.join('saved_model', MODEL_NAME)
TENSORBOARD_SUMMARY_PATH = 'summary'

# reset graph
tf.reset_default_graph()

# random seed
tf.set_random_seed(21)
np.random.seed(21)

# generate some random data with some noise
X = 2 * np.random.rand(100, 1)
y = 8 + 3 * X + np.random.randn(100, 1)

# plot X and y
plt.figure(figsize=(15, 5))
plt.scatter(X, y, marker='x')
plt.xlabel('X')
plt.ylabel('y', rotation=0)
plt.show()


# function to change data type
def convert_dtype(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.float32)


# data input pipeline with tf.data API
with tf.name_scope('data_pipeline'):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(convert_dtype)

    iterator = dataset.make_initializable_iterator()
    x_elem, y_elem = iterator.get_next()

# weight and bias
with tf.name_scope('w_b'):
    w = tf.get_variable(
        'weight',
        shape=(1),
        dtype=tf.float32,
        initializer=tf.random_normal_initializer())

    b = tf.get_variable(
        'bias',
        shape=(1),
        dtype=tf.float32,
        initializer=tf.random_normal_initializer())

# prediction
with tf.name_scope('prediction'):
    y_pred = w * x_elem + b

# loss and optimizer
with tf.name_scope('training'):
    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(labels=y_elem, predictions=y_pred)

    with tf.name_scope('optimizer'):
        opt = tf.train.GradientDescentOptimizer(LEARING_RATE).minimize(loss)

# initialize variables
init = tf.global_variables_initializer()

# persist model
saver = tf.train.Saver()

# save loss after each epoch
loss_history = list()

# session to run graph
with tf.Session() as sess:
    sess.run(init)

    # save summary to visualize the graph in tensorboard
    writer = tf.summary.FileWriter(TENSORBOARD_SUMMARY_PATH, sess.graph)

    print('Before: ', sess.run([w, b]))

    for epoch in range(EPOCH):
        sess.run(iterator.initializer)
        loss_ = list()

        while True:
            try:
                _, loss_value = sess.run([opt, loss])
                loss_.append(loss_value)
            except tf.errors.OutOfRangeError:
                break

        loss_history.append(np.mean(loss_))

        # log loss after 5 epocs and save model
        if epoch % 5 == 0:
            print('Epoch {} loss: {}'.format(epoch, np.mean(loss_)))

            saver.save(sess, MODEL_SAVE_PATH, global_step=epoch)

    writer.close()

    print('After: ', sess.run([w, b]))

# plot loss_history
plt.figure(figsize=(15, 5))
plt.plot(loss_history, 'r-')
plt.xlabel('Epochs')
plt.ylabel('loss', rotation=0)
plt.show()
