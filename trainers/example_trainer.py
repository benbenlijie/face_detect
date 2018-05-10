from base.base_train import BaseTrain
import sys
import numpy as np
import tensorflow as tf
import time


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, config):
        super(ExampleTrainer, self).__init__(sess, model, config)
        self.best_score = float('inf')

    def train(self):
        tf.logging.info("Start to Train")
        if self.model.init_op is not None:
            self.model.init_op(self.sess)
        self.model.load(self.sess)
        for i in range(self.config.num_epochs):
            self.model.init_data_loader(self.sess, train=True)
            self.model.init_data_loader(self.sess, train=False)
            self.val_metric = []
            start_time = time.time()
            while True:
                try:
                    loss, step = self.train_step()
                    self.log_step(loss, step)
                except tf.errors.OutOfRangeError as e:
                    elapsed_time = time.time() - start_time
                    tf.logging.info("Train epoch {} finished. Cost time {}".format(i+1, elapsed_time))
                    print("\nTrain epoch {} finished".format(i + 1))
                    break
            self.model.save(self.sess)
            train_loss = np.mean(self.val_metric)
            # test on val set
            start_time = time.time()
            self.val_metric = []
            self.model.init_data_loader(self.sess, train=True)
            self.model.init_data_loader(self.sess, train=False)
            while True:
                try:
                    self.eval_step()
                except tf.errors.OutOfRangeError as e:
                    break
            elapsed_time = time.time() - start_time
            if len(self.val_metric) > 0:
                self.val_metric = np.mean(self.val_metric, axis=0)
                tf.logging.info("Epoch {}: train loss: {}; val loss: {}; val nme: {}; cost time: {}"
                                .format(i+1, train_loss, self.val_metric[0], self.val_metric[1], elapsed_time))
                print("Epoch {}: train loss: {}; val loss: {}; val nme: {}; cost time: {}"
                      .format(i + 1, train_loss, self.val_metric[0], self.val_metric[1], elapsed_time))
            else:
                print("failed to calculate ")

    def train_step(self):
        global_step, loss, _ = self.sess.run([self.model.global_step, self.model.loss_op, self.model.train_op])
        self.val_metric.append(loss)
        if global_step % self.config.saveInter == 0:
            if self.best_score > loss:
                self.model.save(self.sess)
                self.best_score = loss
        return loss, global_step

    def eval_step(self):
        val_loss, val_nme = self.sess.run(
            [self.model.val_loss, self.model.val_nme])
        # print([val_loss, val_nme])
        self.val_metric.append([val_loss, val_nme])

    def log_step(self, loss, step):
        # sys.stdout.write("step {}: total loss {}, secs/step {}\r".format(step, loss, elapsed_time))
        # sys.stdout.flush()
        print("step {}: total loss {}".format(step, loss))
        if step > 50:
            summary_str = self.sess.run(self.model.summary_op)
            self.model.summary.add_summary(summary_str, step)
            self.model.summary.flush()

