# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:30:41 2019

@author: sahil210695

1. Load exported SavedModel and do prediction
"""
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
'''
Loading SAVED_MODEL in python script
'''

MODEL_NAME = 'linear'
VERSION = 1
MODEL_SAVE_PATH = 'saved_model'
SERVE_PATH = 'serve/{}/{}'.format(MODEL_NAME, VERSION)
LATEST_CHECKPOINT = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
META_GRAPH_PATH = '{}.meta'.format(LATEST_CHECKPOINT)

tf.reset_default_graph()

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               SERVE_PATH)
    graph = tf.get_default_graph()
    m_i = graph.get_tensor_by_name('data_pipeline/IteratorGetNext:0')
    output = graph.get_tensor_by_name("prediction/add:0")

    print(sess.run(output, feed_dict={m_i: [6]}))
