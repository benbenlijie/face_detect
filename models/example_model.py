import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import os
import cv2
import numpy as np

from base.base_model import BaseModel
slim = tf.contrib.slim


class ExampleModel(BaseModel):
    def __init__(self, config, data_loader):
        super(ExampleModel, self).__init__(config, data_loader)

    def _build_model(self, inputs, is_training=True):
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
            logits, endpoints = mobilenet_v2.mobilenet(inputs, num_classes=self.config.num_outputs)
        ema = tf.train.ExponentialMovingAverage(0.999)
        self.mobile_net_vars = [var for var in tf.trainable_variables() if var.name.startswith("Mobilenet") and
                                "Logits" not in var.name]
        return logits, endpoints

    def _build_train_model(self):

        inputs, targets = self.data_loader.get_data()
        # inputs = tf.cast(inputs, tf.float32) / 128. - 1
        inputs = tf.reshape(inputs, (-1, self.config.inputHeight, self.config.inputWidth, 3))

        logits, endpoints = self._build_model(inputs, True)

        with tf.variable_scope("face_keypoint"):
            self.output = tf.layers.dense(logits, self.config.num_outputs, kernel_initializer=tf.contrib.layers.xavier_initializer())

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
        self.train_input = inputs
        self.train_target = targets

    def _build_evaluate_model(self):
        inputs, targets, bbox, scale, originAnnotation, origin_image, image_size = self.data_loader.get_data(False)
        # inputs = tf.cast(inputs, tf.float32) / 128. - 1
        # origin_image = tf.cast(origin_image, tf.float32) / 128. - 1
        inputs = tf.reshape(inputs, (-1, self.config.inputHeight, self.config.inputWidth, 3))
        tf.get_variable_scope().reuse_variables()

        logits, endpoints = self._build_model(inputs, False)

        with tf.variable_scope("face_keypoint", reuse=True):
            val_output = tf.layers.dense(logits, self.config.num_outputs, kernel_initializer=tf.contrib.layers.xavier_initializer())

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
        # tf.summary.image("val_image", inputs)
        # tf.summary.scalar("val_loss", self.val_loss)
        # tf.summary.scalar("nme", self.val_nme)

    def init_op(self, sess):
        self.summary_op = tf.summary.merge_all()
        self.data_loader.init_data_loader(sess, True)
        self.data_loader.init_data_loader(sess, False)
        sess.run(tf.global_variables_initializer())

    def init_data_loader(self, sess, train=True):
        self.data_loader.init_data_loader(sess, train)

    def _loss(self, outputs, targets):
        outputs = tf.reshape(outputs, (-1, self.config.num_outputs // 2, 2))
        outputs = tf.cast(outputs, tf.float32)

        interocular_distance = tf.norm(
            tf.reduce_mean(targets[:, 36:42, :], axis=1) -
            tf.reduce_mean(targets[:, 42:48, :], axis=1), 2, axis=-1)

        output_target_distance = tf.add(tf.norm(
            tf.reduce_mean(outputs[:, 36:42, :], axis=1) -
            tf.reduce_mean(targets[:, 36:42, :], axis=1), 2, axis=-1),
            tf.norm(
                tf.reduce_mean(outputs[:, 42:48, :], axis=1) -
                tf.reduce_mean(targets[:, 42:48, :], axis=1), 2, axis=-1))

        loss_op = tf.reduce_mean(tf.reduce_mean(
            tf.norm(outputs - targets, 2, axis=2),
            axis=1) / interocular_distance + output_target_distance)
        return loss_op

    def _nme_loss(self, outputs, targets):
        interocular_distance = tf.norm(
            tf.reduce_mean(targets[:, 36:42, :], axis=1) -
            tf.reduce_mean(targets[:, 42:48, :], axis=1), 2, axis=-1)
        outputs = tf.reshape(outputs, (-1, self.config.num_outputs // 2, 2))
        outputs = tf.cast(outputs, tf.float32)
        loss_op = tf.reduce_mean(tf.reduce_sum(
            tf.norm(outputs - targets, 2, axis=2),
            axis=1) / (interocular_distance * self.config.num_outputs / 2))
        return loss_op

    def load(self, sess):
        if self.config.fine_tune:
            latest_checkpoint = os.path.join(self.config.checkpoint_dir, self.config.init_checkpoint)
            if latest_checkpoint:
                tf.logging.info("Loading model checkpoint from {}".format(latest_checkpoint))
                saver = tf.train.Saver(self.mobile_net_vars)
                saver.restore(sess, latest_checkpoint)
                tf.logging.info("Model loaded")
        else:
            super(ExampleModel, self).load(sess)

    def _save(self, sess, global_step=None):
        tf.logging.info("Saving model...")
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), global_step)
        tf.logging.info("Model saved")

    def save_val_image(self, image_arr, annotation, file_name, size=None):
        annotation = np.reshape(annotation, (len(annotation) // 2, 2))
        image_arr = (image_arr + 1) * 128.
        image = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
        if size is not None:
            image = cv2.resize(image, tuple(size))
        for i  in range(len(annotation)):
            point = annotation[i]
            if i < (self.config.num_outputs // 2):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.circle(image, center=tuple(point), color=color, radius=2, thickness=2)
        write_path = os.path.join(self.config.summary_dir, file_name)
        cv2.imwrite(write_path, image)
