import tensorflow as tf

from data_loader import *
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.utils import get_args
import matplotlib.pyplot as plt

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    config.num_epochs = None
    image_loader = DatasetLoader(config, True)
    input = image_loader.get_data()
    # model = ExampleModel(config, image_loader)
    # model.init_train_model()
    import cv2
    with tf.Session() as sess:
        for i in range(4):
            output = sess.run(input)
            img = cv2.cvtColor(output[0], cv2.COLOR_BGR2RGB)
            for point in output[1]:
                cv2.circle(img, (int(point[0]), int(point[1])), 2, (255, 255, 0))
            plt.imshow(img)
            plt.show()
        # trainer = ExampleTrainer(sess, model, config)
        # trainer.train()


if __name__ == '__main__':
    main()
