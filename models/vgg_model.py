import tensorflow as tf
from nets import vgg
import os
import cv2
import numpy as np

from models.example_model import ExampleModel
slim = tf.contrib.slim


class VGGModel(ExampleModel):
    def __init__(self, config, data_loader):
        super(VGGModel, self).__init__(config, data_loader)

    def _build_train_model(self):

        inputs, targets = self.data_loader.get_data()
        # inputs = tf.cast(inputs, tf.float32) / 128. - 1
        inputs = tf.reshape(inputs, (-1, self.config.inputHeight, self.config.inputWidth, 3))
        logits, _ = vgg.vgg_19(inputs, 1000, is_training=True)

        with tf.variable_scope("face_keypoint"):
            self.output = tf.layers.dense(logits, self.config.num_outputs, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.mobile_net_vars = [var for var in tf.trainable_variables() if var.name.startswith("vgg")]
        self.save_variables = tf.global_variables()
        self.loss_op = self._loss(self.output, targets)
        tf.summary.scalar("loss", self.loss_op)
        slim.summarize_tensors(self.save_variables)

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, self.config.lr_step, 0.66,
                                                   name="learning_rate")
        tf.summary.image("train_image", inputs)
        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, self.global_step)

    def _build_evaluate_model(self):
        inputs, targets, bbox, scale, originAnnotation, origin_image, image_size = self.data_loader.get_data(False)
        # inputs = tf.cast(inputs, tf.float32) / 128. - 1
        # origin_image = tf.cast(origin_image, tf.float32) / 128. - 1
        inputs = tf.reshape(inputs, (-1, self.config.inputHeight, self.config.inputWidth, 3))
        tf.get_variable_scope().reuse_variables()
        logits, _ = vgg.vgg_19(inputs, 1000, is_training=False)

        with tf.variable_scope("face_keypoint", reuse=True):
            val_output = tf.layers.dense(logits, self.config.num_outputs)

        reshaped_output = tf.reshape(val_output, (-1, self.config.num_outputs//2, 2))
        reshaped_output = tf.transpose(reshaped_output, (1, 0, 2))

        wh_factor = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]], dtype=np.float32)
        tf_wh = tf.matmul(bbox, wh_factor.T)

        self.val_annotation = tf.transpose((reshaped_output + 1.) / 2. *
                                           np.array([self.config.inputWidth, self.config.inputHeight]), (1, 0, 2))
        self.val_origin_output = tf.transpose(
            (reshaped_output + 1.) / 2. * tf_wh + bbox[:, :2], (1, 0, 2))
#        / scale) + offset, (1, 0, 2))

        self.val_input = inputs
        self.val_target = targets

        self.val_target_annotation = tf.transpose(
            tf.transpose((targets + 1.) / 2., (1, 0, 2)) * np.array([self.config.inputWidth, self.config.inputHeight]),
            (1, 0, 2))
        self.val_originAnnotation = originAnnotation
        self.val_loss = self._loss(val_output, targets)
        self.val_nme = self._nme_loss(self.val_origin_output, self.val_originAnnotation)
        self.val_image = origin_image
        self.val_image_size = image_size
        tf.summary.image("val_image", inputs)
        tf.summary.scalar("val_loss", self.val_loss)
        tf.summary.scalar("nme", self.val_nme)
