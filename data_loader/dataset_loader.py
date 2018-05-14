import tensorflow as tf
from bunch import Bunch
import numpy as np
import os
import cv2

class DatasetLoader(object):
    filp_mapping = np.array(
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29,
         30, 35, 34, 33, 32, 31,
         45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61,
         60, 67, 66, 65])

    def __init__(self, config: Bunch, shuffle=True):
        """

        :param config: {"dataFile", "bboxFile", "imageFolder", "annotationFolder",
                        "inputHeight", "inputWidth", "margin"[left, top, right, bottom]}
        :param shuffle:
        """
        self._init_config(config)
        self.bboxInfos = dict()
        self.infos = dict()
        self.shuffle = shuffle
        self.dataset = self._create_dataset(self.bboxFile, self.dataFile, self.imageFolder, self.annotationFolder)
        self.valDataset = self._create_dataset(
            self.valBboxFile, self.valDataFile, self.valImageFolder, self.valAnnotationFolder, train=False)
        self.data_iterator = dict()

    def _init_config(self, config: Bunch):
        self.config = config
        self.bboxFile = self.config.bboxFile
        self.dataFile = self.config.dataFile
        self.imageFolder = self.config.imageFolder
        self.annotationFolder = self.config.annotationFolder

        self.valBboxFile = self.config.valBboxFile
        self.valDataFile = self.config.valDataFile
        self.valImageFolder = self.config.valImageFolder
        self.valAnnotationFolder = self.config.valAnnotationFolder

    def _create_dataset(self, bboxFile, dataFile, imageFolder, annotationFolder, train=True):
        with open(bboxFile, "r") as f:
            textLines = f.readlines()[1:]
        bboxTexts = [line.replace("\n", "").split(",") for line in textLines]

        info_key = self._get_train_key(train)
        infos = {}
        infos["bboxInfos"] = {line[0]: list(map(int, line[1:])) for line in bboxTexts}
        infos["imageFolder"] = imageFolder
        infos["annotationFolder"] = annotationFolder

        if train:
            output_type = [tf.float32, tf.float32]
        else:
            output_type = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32]


        dataset = tf.data.TextLineDataset(dataFile)
        # with open(dataFile, "r") as f:
        #     dataLines = f.readlines()
        # dataLines = [line.strip() for line in dataLines]
        # print("data file: ", info_key, dataFile, len(dataLines))
        # dataset = tf.data.Dataset.from_tensor_slices(dataLines)

        # dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #     map_func = lambda fileName: tf.py_func(
        #         self.prepareInput, [fileName, train], output_type),
        #     batch_size = self.config.batch_size
        # ))
        dataset = dataset.prefetch(buffer_size=self.config.batch_size * 100)
        dataset = dataset.map(
            lambda fileName: tf.py_func(
                self.prepareInput, [fileName, train], output_type),
            num_parallel_calls=32)

        if self.shuffle and train:
            dataset = dataset.shuffle(self.config.batch_size * 10)
        dataset = dataset.batch(self.config.batch_size)

        self.infos[info_key] = infos
        return dataset

    def get_data(self, train=True):
        info_key = self._get_train_key(train)
        if train:
            iterator = self.dataset.make_initializable_iterator()
        else:
            iterator = self.valDataset.make_initializable_iterator()
        self.data_iterator[info_key] = iterator
        return iterator.get_next()

    def init_data_loader(self, sess, train=True):
        info_key = self._get_train_key(train)
        # if info_key in self.data_iterator:
        iterator = self.data_iterator[info_key]
        sess.run(iterator.initializer)

    def prepareInput(self, inFileName, train):
        fileName = inFileName.decode("UTF-8")
        info_key = self._get_train_key(train)
        # print(fileName, info_key)
        infos = self.infos[info_key]
        imageFolder = infos["imageFolder"]
        annotationFolder = infos["annotationFolder"]
        filePath = os.path.join(imageFolder, fileName)

        # read raw image
        image = cv2.imread(filePath, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        # prepare bbox data
        bbox = infos["bboxInfos"][fileName]
        bbox = [np.clip(val[0], 0, val[1]) for val in zip(bbox, [width, height] * 2)]

        # prepare annotation data
        annotationPath = os.path.join(annotationFolder, "{}.pts".format(fileName.split(".")[0]))
        with open(annotationPath, "r") as f:
            annotationLines = f.readlines()
        # ignore useless lines
        annotationLines = annotationLines[3:-1]
        annotation = [[float(value[0]),
                       float(value[1])]
                      for value in [line.split(" ") for line in annotationLines]]

        annotation = np.array(annotation)

        if train:
            # enhance image with random flip and random rotate
            # randomFlip image+bbox+annotation
            image, bbox, annotation = self._randomFlip(image, bbox, annotation)
            # randomRotate image+bbox+annotation
            image, bbox, annotation = self._randomRotate(image, bbox, annotation)

        # crop face
        bbox = self.adjustBbox(bbox, [width, height])
        xmin, ymin, xmax, ymax = bbox
        face = image[ymin:ymax, xmin:xmax]
        originWidth, originHeight = xmax - xmin, ymax - ymin

        xScale, yScale = 1. / originWidth, 1. / originHeight
        try:
            face = cv2.resize(face, (self.config.inputWidth, self.config.inputHeight))
        except:
            print ("error!!!", fileName, train, face.shape, infos["bboxInfos"][fileName], bbox)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 128. - 1.
        face = face.astype(np.float32)

        bbox = np.array(bbox, np.float32)
        scale = np.array([xScale, yScale], np.float32)
        originAnnotation = np.array(annotation, np.float32)
        # adjust annotation and normalized to -1~1 value
        # annotation = np.array([[(point[0] - xmin) * xScale * 2. - 1., (point[1] - ymin) * yScale * 2. - 1.]
        #                        for point in annotation], dtype=np.float32)
        annotation = self.normalize_annotation(annotation, bbox)
        if train:
            return face, annotation
        else:
            resize_origin_image = 1000
            origin_image = cv2.resize(image, (resize_origin_image, resize_origin_image))
            origin_image_size = np.array([width, height], np.int32)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
            origin_image = origin_image / 128. - 1
            origin_image = origin_image.astype(np.float32)
            # provide offset, [xScale, yScale] to translate the output to origin axis.
            return face, annotation, bbox, scale, originAnnotation, \
                   origin_image, origin_image_size

    def normalize_annotation(self, annotation, bbox):
        xmin, ymin, xmax, ymax = bbox
        originWidth, originHeight = xmax - xmin, ymax - ymin
        return np.array([[(point[0] - xmin) * 2. / originWidth - 1., (point[1] - ymin) * 2. / originHeight - 1.]
                               for point in annotation], dtype=np.float32)

    def unnormalize_annotation(self, annotation, bbox):
        xmin, ymin, xmax, ymax = bbox
        originWidth, originHeight = xmax - xmin, ymax - ymin
        np.array([[(point[0] + 1) / 2. * originWidth + xmin, (point[1] + 1) / 2. * originHeight + ymin]
                  for point in annotation], dtype=np.float32)
        pass

    def _randomFlip(self, img, bbox, annotation):
        """

        :param img: [height, width, channel]
        :param bbox:  [xmin, ymin, xmax, ymax]
        :param annotation: [[x, y], ...] shape == [64, 2]
        :return:
        img, bbox, annotation
        """
        # 50% to flip
        if np.random.random() < 0.5:
            return img, bbox, annotation
        height, width, _ = img.shape
        # flip img
        img = cv2.flip(img, 1)
        # flip bbox
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = np.clip(width - xmax, 0, width - 1), np.clip(width - xmin, 0, width - 1)
        bbox = [xmin, ymin, xmax, ymax]
        # flip annotation
        annotation = np.array([[width - point[0], point[1]] for point in annotation])
        # reorder point
        annotation = annotation[DatasetLoader.filp_mapping]

        return img, bbox, annotation

    def _randomRotate(self, img, bbox, annotation, angle=15):
        """

        :param img: [height, width, channel]
        :param bbox:  [xmin, ymin, xmax, ymax]
        :param annotation: [[x, y], ...] shape == [64, 2]
        :return:
        img, bbox, annotation
        """
        # 33% to rotate
        if np.random.random() < 2/3:
            return img, bbox, annotation
        elif np.random.random() < 0.5:
            angle *= -1
        height, width, channel = img.shape
        rotationMat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

        # rotate img
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        newImg = cv2.warpAffine(img, rotationMat, (width, height), borderValue=avg_color)

        # rotate bbox
        bboxPoints = np.array([[x, y, 1] for x in bbox[::2] for y in bbox[1::2]])
        bboxPoints = rotationMat.dot(bboxPoints.T).T
        newBbox = [np.min(bboxPoints[:, 0]), np.min(bboxPoints[:, 1]), np.max(bboxPoints[:, 0]), np.max(bboxPoints[:, 1])]
        newBbox = [int(np.clip(pos[0], 0, pos[1])) for pos in zip(newBbox, [width, height] * 2)]

        xmin, ymin, xmax, ymax = newBbox
        originWidth, originHeight = xmax - xmin, ymax - ymin
        if originWidth == 0 or originHeight == 0:
            return img, bbox, annotation
        
        # rotate annotation
        annotationMat = np.concatenate((annotation, np.ones((len(annotation), 1))), axis=1)
        annotation = rotationMat.dot(annotationMat.T).T

        return newImg, newBbox, annotation

    def adjustBbox(self, bbox, imgSize):
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax - xmin, ymax - ymin
        newBbox = [ int(np.clip(pos[0] + pos[1] * pos[2], 0, pos[3]-1))
                    for pos in zip(bbox, self.config.margin, [width, height] * 2, imgSize * 2)]

        return newBbox

    def _get_train_key(self, train):
        info_key = "train" if train else "val"
        return info_key