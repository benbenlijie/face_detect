import tensorflow as tf

from data_loader import *
from models.example_model import ExampleModel
from models.vgg_model import VGGModel
from models.custom_model import CustomModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments", e)
        exit(0)

    image_loader = DatasetLoader(config, True)
    if "vgg" in config.init_checkpoint:
        model = VGGModel(config, image_loader)
    else:
        model = CustomModel(config, image_loader)
    model.init_train_model()
    model.init_evaluate_model()

    with tf.Session() as sess:
        trainer = ExampleTrainer(sess, model, config)
        trainer.train()


if __name__ == '__main__':
    main()
