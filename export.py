# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:30:41 2019

@author: sahil210695

1. Export trained model into SavedModel format
"""
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

MODEL_NAME = 'linear'
VERSION = 1
MODEL_SAVE_PATH = 'saved_model'
SERVE_PATH = os.path.join('serve', MODEL_NAME, str(VERSION))
LATEST_CHECKPOINT = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
META_GRAPH_PATH = '{}.meta'.format(LATEST_CHECKPOINT)

with tf.Session() as sess:
    # Get graph from curent session
    graph = tf.get_default_graph()

    # Load meta graph file
    restored_model = tf.train.import_meta_graph(META_GRAPH_PATH)

    # Initialize the loaded graph
    restored_model.restore(sess, LATEST_CHECKPOINT)

    # Fetch tensors from graph which are required for predictions using their names
    inputs = graph.get_tensor_by_name('data_pipeline/IteratorGetNext:0')
    prediction = graph.get_tensor_by_name("prediction/add:0")

    # Build tensor info
    model_input = tf.saved_model.utils.build_tensor_info(inputs)
    model_output = tf.saved_model.utils.build_tensor_info(prediction)

    # Create the model signature definition
    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"x": model_input},
        outputs={"y": model_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_definition
        })

    # Save the model so we can serve it with a model server
    builder.save()

print('Done exporting!')
