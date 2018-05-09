import tensorflow as tf
import os
import cv2
import numpy as np

from models.example_model import ExampleModel
slim = tf.contrib.slim


class CustomModel(ExampleModel):
    def __init__(self, config, data_loader):
        super(CustomModel, self).__init__(config, data_loader)

    def _build_model(self, inputs, is_training=True):
        return self._model(inputs, num_classes=self.config.num_outputs, is_training=is_training)

    def _model(self, inputs,
                   num_classes=1000,
                   is_training=True,
                   dropout_keep_prob=0.5,
                   scope='vgg_cust'):
        with tf.variable_scope(scope, 'vgg_cust', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                with tf.variable_scope("conv1"):
                    net = tf.layers.conv2d(inputs, 96, 7, strides=(2, 2), padding="valid",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name="conv_1")
                    net = tf.nn.relu(net, name="relu_1")
                    net = tf.layers.max_pooling2d(net, 3, 3, name="max_pool_1")
                    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training, name="drop_1")
                with tf.variable_scope("conv2"):
                    net = tf.layers.conv2d(net, 256, 5, name="conv_2",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                    net = tf.nn.relu(net, name="relu_2")
                    net = tf.layers.max_pooling2d(net, 2, 2, name="max_pool_2")
                    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training, name="drop_2")
                with tf.variable_scope("conv3"):
                    net = tf.layers.conv2d(net, 512, 3, name="conv_3",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same")
                    net = tf.nn.relu(net, name="relu_3")
                    net = tf.layers.conv2d(net, 512, 3, name="conv_4",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same")
                    net = tf.nn.relu(net, name="relu_4")
                    net = tf.layers.conv2d(net, 512, 3, name="conv_5",
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same")
                    net = tf.nn.relu(net, name="relu_5")
                    net = tf.layers.max_pooling2d(net, 3, 3, name="max_pool_3")
                    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training, name="drop_3")
                with tf.variable_scope("dense"):
                    net = tf.layers.flatten(net, name="flatten")
                    net = tf.layers.dense(net, 4096, name="dense_6",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
                    net = tf.nn.relu(net, name="relu_6")
                    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training, name="drop_4")
                    net = tf.layers.dense(net, 4096, name="dense_7",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
                    net = tf.nn.relu(net, name="relu_7")
                    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training, name="drop_4")
                    net = tf.layers.dense(net, num_classes, name="output",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return net, end_points