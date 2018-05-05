import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import os

from base.base_model import BaseModel
slim = tf.contrib.slim


class ExampleModel(BaseModel):
    def __init__(self, config, data_loader):
        super(ExampleModel, self).__init__(config, data_loader)

    def _build_train_model(self):

        inputs, targets = self.data_loader.get_data()
        inputs = tf.cast(inputs, tf.float32) / 128. - 1
        inputs = tf.reshape(inputs, (-1, self.config.inputHeight, self.config.inputWidth, 3))
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
            logits, endpoints = mobilenet_v2.mobilenet(inputs)

        ema = tf.train.ExponentialMovingAverage(0.999)

        with tf.variable_scope("face_keypoint"):
            self.output = tf.layers.dense(logits, self.config.num_outputs)

        self.save_variables = tf.trainable_variables()
        self._loss(self.output, targets)
        pass

    def _build_evaluate_model(self):
        pass

    def init_op(self, sess):
        sess.run(tf.global_variables_initializer())

    def _loss(self, outputs, targets):
        interocular_distance = tf.norm(
            tf.reduce_mean(targets[:, 36:42, :], axis=1) -
            tf.reduce_mean(targets[:, 42:48, :], axis=1), 2, axis=-1)
        outputs = tf.reshape(outputs, (-1, self.config.num_outputs // 2, 2))
        outputs = tf.cast(outputs, tf.float64)
        self.loss_op = tf.reduce_mean(tf.reduce_mean(
            tf.norm(outputs - targets, 2, axis=2),
            axis=1) / interocular_distance)
        tf.summary.scalar("loss", self.loss_op)
        tf.logging.info(self.loss_op.shape)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, self.global_step)
        pass

    def __load(self, sess):
        latest_checkpoint = os.path.join(self.config.checkpoint_dir, self.config.init_checkpoint)
        if latest_checkpoint:
            tf.logging.info("Loading model checkpoint from {}".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            tf.logging.info("Model loaded")
