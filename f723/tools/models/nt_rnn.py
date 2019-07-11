import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

from f723.tools.dataset.features import FeatureVector
from f723.tools.dataset.utils import split_feature_vector


class NucleotideRNN:
    def __init__(self, neighbours_num, batch_size, epoch_size, hidden_size, discriminator_size,
                 num_epochs, threshold=0.5, validation_rate=None):
        self.validation_rate = validation_rate
        self.epoch_size = epoch_size
        self.num_epochs = num_epochs
        self.neighbours_num = neighbours_num
        self.seq_len = 2 * self.neighbours_num + 1
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.discriminator_size = discriminator_size
        self.threshold = threshold

    def split_train_validation(self, feature_vector):
        if self.validation_rate is None:
            train_data = feature_vector
            validation_data = None
        else:
            splitter = GroupShuffleSplit(train_size=1 - self.validation_rate, test_size=self.validation_rate)
            train_index, test_index = next(splitter.split(
                feature_vector.features, feature_vector.target, feature_vector.pdb_ids))

            train_data, validation_data = split_feature_vector(feature_vector, [train_index, test_index])

        return train_data, validation_data

    def build_graph(self, train_data):
        tf.reset_default_graph()

        self.features_placeholder = tf.placeholder(
            tf.float32, shape=(self.batch_size, self.seq_len, train_data.features.shape[2]))
        self.target_placeholder = tf.placeholder(
            tf.bool, shape=(self.batch_size, self.seq_len))

        cells = [tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) for _ in range(2)]
        (outputs_fw, outputs_bw), (last_fw, last_bw) = tf.nn.bidirectional_dynamic_rnn(
            cells[0], cells[1], self.features_placeholder, dtype=tf.float32)

        outputs = outputs_fw + outputs_bw
        normalized_outputs = tf.nn.l2_normalize(outputs, axis=2)

        discriminator = tf.get_variable(
            name='discriminator',
            initializer=tf.random_normal_initializer(0, 1),
            shape=(self.hidden_size, self.discriminator_size))
        normalized_discriminator = tf.nn.l2_normalize(discriminator, axis=0)

        scores = tf.einsum('mnj,ji->mni', normalized_outputs, normalized_discriminator)

        self.alpha = tf.Variable(1, dtype=tf.float32)
        self.proba = tf.math.sigmoid(self.alpha * tf.reduce_max(scores, axis=2))

        self.loss = tf.losses.log_loss(self.target_placeholder, self.proba)

        self.learning_rate = tf.Variable(0.01)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def batch_generator(self, train_data, num_batches):
        index = np.arange(len(train_data.features))

        for _ in range(num_batches):
            batch_index = np.random.choice(index, size=self.batch_size, replace=False)

            yield train_data.features[batch_index], train_data.target[batch_index]

    def train(self, train_data, validation_data):
        self.sess = tf.Session()
        self.sess.run(tf.initializers.global_variables())

        loss_history = []

        for _ in range(self.num_epochs):
            for batch_features, batch_target in self.batch_generator(train_data, self.epoch_size):
                feed_dict = {
                    self.features_placeholder: batch_features,
                    self.target_placeholder: batch_target
                }
                _, batch_loss, batch_alpha = self.sess.run(
                    [self.train_step, self.loss, self.alpha], feed_dict=feed_dict)

                loss_history.append(batch_loss)

            print('loss: {}; alpha: {}'.format(np.mean(loss_history[-self.epoch_size:]), batch_alpha))

            plt.figure(figsize=(20, 10))

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('precision', fontsize=23)
            plt.ylabel('recall', fontsize=23)

            target, proba = self.infer_train(train_data)
            precision, recall, thresholds = precision_recall_curve(target.ravel(), proba.ravel())
            plt.plot(precision, recall, c='r', label='train')

            if validation_data is not None:
                target, proba = validation_data.target, self.infer(validation_data.features)
                precision, recall, thresholds = precision_recall_curve(target.ravel(), proba.ravel())
                plt.plot(precision, recall, c='b', label='validation')

            dots = np.arange(0, 1, 0.1)
            plt.scatter(dots, dots)

            plt.legend(fontsize=15)
            plt.show()

    def infer_train(self, train_data):
        target, proba = [], []

        for batch_features, batch_target in self.batch_generator(train_data, self.epoch_size // 10):
            feed_dict = {
                self.features_placeholder: batch_features
            }
            batch_proba = self.sess.run(self.proba, feed_dict=feed_dict)

            target.extend(batch_target)
            proba.extend(batch_proba)

        return np.array(target), np.array(proba)

    def infer(self, features):
        batch_slices = [slice(index * self.batch_size, (index + 1) * self.batch_size)
                        for index in range(features.shape[0] // self.batch_size)]
        batch_slices.append(slice(-self.batch_size, None))
        predicted_proba = np.zeros(features.shape[:2], dtype=np.float32)

        for sl in batch_slices:
            feed_dict = {self.features_placeholder: features[sl]}

            predicted_proba[sl] = self.sess.run(self.proba, feed_dict=feed_dict)

        return predicted_proba

    def fit(self, feature_vector):
        train_data, validation_data = self.split_train_validation(feature_vector)

        self.build_graph(train_data)

        self.train(train_data, validation_data)

    def predict_proba(self, features):
        return self.infer(features)[:, self.neighbours_num]
