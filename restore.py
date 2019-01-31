# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:30:41 2019

@author: sahil210695

1. Restore saved model from latest checkpoints
"""

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

MODEL_NAME = 'linear'
MODEL_SAVE_PATH = 'saved_model'
LATEST_CHECKPOINT = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
META_GRAPH_PATH = '{}.meta'.format(LATEST_CHECKPOINT)

with tf.Session() as sess:
    # get graph from curent session
    graph = tf.get_default_graph()

    # load meta graph file
    restored_model = tf.train.import_meta_graph(META_GRAPH_PATH)

    # initialize the loaded graph
    restored_model.restore(sess, LATEST_CHECKPOINT)

    # fetch tensors from graph which are required for predictions using their names
    x_elem = graph.get_tensor_by_name('data_pipeline/IteratorGetNext:0')
    y_pred = graph.get_tensor_by_name("prediction/add:0")

    random_input = np.random.randint(10, size=(1))

    # prediction with random input
    output = sess.run(y_pred, feed_dict={x_elem: random_input})

    print('Input: {}'.format(random_input))
    print('Output: {}'.format(output))
