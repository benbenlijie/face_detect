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

        self.model.load(self.sess)
        for i in range(self.config.num_epochs):
            if self.model.init_op is not None:
                self.model.init_op(self.sess)
            self.val_metric = []
            try:
                while True:
                    start_time = time.time()
                    self.train_step()
                    elapsed_time = time.time() - start_time
                    self.log_step(elapsed_time)
            except tf.errors.OutOfRangeError as e:
                tf.logging.info("Train epoch {} finished".format(i+1))
            finally:
                self.model.save(self.sess)
            train_loss = np.mean(self.val_metric)
            # test on val set
            start_time = time.time()
            self.val_metric = []
            try:
                while True:
                    self.eval_step()
                pass
            except tf.errors.OutOfRangeError as e:
                pass
            elapsed_time = time.time() - start_time
            self.val_metric = np.mean(self.val_metric, axis=0)
            tf.logging.info("Epoch {}: train loss: {}; val loss: {}; val nme: {}; cost time: {}"
                            .format(i+1, train_loss, self.val_metric[0], self.val_metric[1], elapsed_time))

    def train_step(self):
        global_step, loss, _ = self.sess.run([self.model.global_step, self.model.loss_op, self.model.train_op])
        self.val_metric.append(loss)
        if global_step % self.config.saveInter == 0:
            val_loss, val_input = self.sess.run([self.model.val_loss, self.model.val_input])
            if self.best_score > val_loss:
                self.model.save(self.sess)
                self.best_score = val_loss

    def eval_step(self):
        val_loss, val_nme = self.sess.run(
            [self.model.val_loss, self.model.val_nme])
        self.val_metric.append([val_loss, val_nme])

    def log_step(self, elapsed_time=0):
        loss, step = self.sess.run([self.model.loss_op, self.model.global_step])
        sys.stdout.write("step {}: total loss {}, secs/step {}\r".format(step, loss, elapsed_time))
        sys.stdout.flush()
        if step > 50:
            summary_str = self.sess.run(self.model.summary_op)
            self.model.summary.add_summary(summary_str, step)
            self.model.summary.flush()

