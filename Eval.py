import tensorflow as tf

from data_loader import *
from models.mobilenet_model import MobileNetModel
from models.vgg_model import VGGModel
from models.custom_model import CustomModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.utils import get_args
import numpy as np

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments", e)
        exit(0)

    image_loader = DatasetLoader(config, False)
    if "mobilenet" in config.init_checkpoint:
        model = MobileNetModel(config, image_loader)
    else:
        model = CustomModel(config, image_loader)
    model.init_train_model()
    model.init_evaluate_model()

    with tf.Session() as sess:
        if model.init_op is not None:
            model.init_op(sess)
        model.load(sess)
        for i in range(20):
            img, output, target = sess.run([model.val_input, model.val_annotation, model.val_target_annotation])

            annotations = np.concatenate((output[0].flatten(), target[0].flatten()))
            # print(output.shape, output[0])
            # print(target.shape, target[0])
            # print(annotations)
            model.save_val_image(img[0], annotations, "val_test_{}.jpg".format(i+1))
            # image, target = sess.run([model.train_input, model.train_target])
            # image = image[0]
            # target = target[0]
            # # image = (image + 1.) * 128.
            # target = (target + 1.) / 2. * config.inputWidth
            # # target = np.reshape(target, (len(target) // 2, 2))
            # # print(target.shape)
            # model.save_val_image(image, target.flatten(), "train_test_{}.jpg".format(i+1))


if __name__ == '__main__':
    main()
