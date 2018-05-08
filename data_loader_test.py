import tensorflow as tf
from utils.config import process_config
from utils.utils import get_args
from data_loader import *

def main():
    try:
        args = get_args()
        config = process_config("configs/test.json")

    except:
        print("missing or invalid arguments")
        exit(0)

    image_loader = DatasetLoader(config, False)
    train_img, train_anno = image_loader.get_data()
    val_img, val_anno, val_offset, val_scale, val_origin_annotation, val_origin_image, val_origin_size = \
        image_loader.get_data(False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img, anno = sess.run([train_img, train_anno])
        print("[train] img size:", img.shape, "\nannotations:", anno)
        img, anno, offset, scale, origin_anno, origin_img, origin_size = sess.run(
            [val_img, val_anno, val_offset, val_scale, val_origin_annotation, val_origin_image, val_origin_size])
        print("[val] img size:", img.shape, "\nannotations:", anno, "\noffset", offset,
              "\norigin annotation:", origin_anno,
              "\norigin img size:", origin_img.shape, origin_size)
    pass

if __name__ == '__main__':
    main()