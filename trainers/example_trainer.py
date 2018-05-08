from base.base_train import BaseTrain
import sys
import numpy as np


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, config):
        super(ExampleTrainer, self).__init__(sess, model, config)
        self.best_score = float('inf')

    def train_step(self):
        self.sess.run(self.model.train_op)
        global_step = self.sess.run(self.model.global_step)
        if global_step % self.config.saveInter == 0:
            val_loss, val_input = self.sess.run([self.model.val_loss, self.model.val_input])
            if self.best_score > val_loss:
                self.model.save(self.sess)
                self.best_score = val_loss


    def log_step(self, elapsed_time=0):
        loss, step = self.sess.run([self.model.loss_op, self.model.global_step])
        sys.stdout.write("step {}: total loss {}, secs/step {}\r".format(step, loss, elapsed_time))
        sys.stdout.flush()
        summary_str = self.sess.run(self.model.summary_op)
        self.model.summary.add_summary(summary_str, step)
        self.model.summary.flush()
        if step % self.config.logInter == 0:
            val_input, val_output, val_target = self.sess.run(
                [self.model.val_input, self.model.val_annotation, self.model.val_target_annotation])
            if val_input.ndim == 4:
                for i in range(len(val_input)):
                    image_arr = val_input[i]
                    annotation = val_output[i].flatten()
                    target_anno = val_target[i].flatten()
                    annotation = np.concatenate((annotation, target_anno))
                    file_name = "val_output_{}_{}.jpg".format(step, i)
                    self.model.save_val_image(image_arr, list(map(int, annotation)), file_name)
            val_image, val_origin_output, val_origin_annotation, val_image_size = self.sess.run([
                self.model.val_image, self.model.val_origin_output, self.model.val_originAnnotation, self.model.val_image_size
            ])
            if val_image.ndim == 4:
                for i in range(len(val_image)):
                    image_arr = val_image[i]
                    annotation = val_origin_output[i].flatten()
                    target_anno = val_origin_annotation[i].flatten()
                    annotation = np.concatenate((annotation, target_anno))
                    file_name = "val_origin_output_{}_{}.jpg".format(step, i)
                    self.model.save_val_image(image_arr, list(map(int, annotation)), file_name, val_image_size[i])
