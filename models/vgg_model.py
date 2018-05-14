import tensorflow as tf
from nets import vgg
import os
import cv2
import numpy as np

from models.mobilenet_model import MobileNetModel
slim = tf.contrib.slim


class VGGModel(MobileNetModel):
    def __init__(self, config, data_loader):
        super(VGGModel, self).__init__(config, data_loader)

    def _build_model(self, inputs, is_training=True):
        result = vgg.vgg_19(inputs, 1000, is_training=is_training)
        self.mobile_net_vars = [var for var in tf.trainable_variables() if var.name.startswith("vgg")]
        return result
