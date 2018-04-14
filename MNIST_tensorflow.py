import tensorflow as tf
from Model import Model
from tensorflow.examples.tutorials.mnist import input_data

import logging
import os
import time
logger = logging.getLogger("MNIST")


"""
Main File for MNIST Implementation using Tensorflow 
"""
class Config(object):
    """Holds model hyperparams and data information.
    """
    n_features = 784
    n_classes = 10
    dropout = 0.5  # (p_drop in the handout)
    hidden_size = 300
    batch_size = 1000
    n_epochs = 10
    lr = 0.0005
    beta = 0.01 # to be used in regularization
    extension_on =False


class MNISTModel(Model):

    def __init__(self, config):
        with tf.variable_scope("MNIST_Model"):
            self.config = config
            self.build_graph()

    def build_graph(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.acc = self.calc_accuracy(self.pred,self.labels_placeholder)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        cn = Config()
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,cn.n_features])
        self.labels_placeholder = tf.placeholder(dtype=tf.int32,shape=[None,cn.n_classes])
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):

        feed_dict = dict()
        feed_dict[self.input_placeholder]= inputs_batch

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_prediction_op(self):

        cn = Config()

        x = self.input_placeholder # [ ? , 784]
        W = tf.get_variable("Weights_Layer1", dtype=tf.float32,shape=(cn.n_features, cn.hidden_size), initializer=tf.contrib.layers.xavier_initializer()) # [784, 300]
        b1 = tf.get_variable("bias_Layer1",shape=(1, cn.hidden_size), initializer=tf.zeros_initializer()) #[1, 300]

        h = tf.matmul(x, W)+b1

        U = tf.get_variable("Weights_Layer2", shape=(cn.hidden_size, cn.n_classes), initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("bias_Layer2",shape=(1, cn.n_classes), initializer=tf.zeros_initializer)
        pred = tf.nn.softmax(tf.matmul(h, U) + b2)

        return pred

    def add_loss_op(self, pred):

        out1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder,logits=pred)
        loss = tf.reduce_mean(out1)

        return loss

    def add_training_op(self, loss):

        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch):

        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss, accuracy = sess.run([self.train_op, self.loss,self.acc], feed_dict=feed)

        return loss,accuracy

    def predict_on_batch(self, sess, inputs_batch, gold_labels):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=gold_labels)  # we are not passing any labels as we want to predict not train
        predictions, accuracy = sess.run([self.pred, self.acc], feed_dict=feed)

        return predictions, accuracy

    def calc_accuracy(self, pred, labels):

        correct_predictions = tf.equal(tf.arg_max(pred, 1), tf.arg_max(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

        return accuracy

    def run_epoch(self, sess, saver, train, dev):

        num_examples = train.images.shape[0]
        batch_size = self.config.batch_size
        num_batches = num_examples / batch_size

        loss_batches = 0
        for i in range(55): #num_batches is float bug here
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_train = train.images[start_idx:end_idx, :]
            y_train = train.labels[start_idx:end_idx, :]

            loss, accuracy = self.train_on_batch(sess, x_train, y_train)
            loss_batches = loss_batches + loss

        x_dev = dev.images
        y_dev = dev.labels

        pred_dev, dev_accuracy = self.predict_on_batch(sess, x_dev, y_dev)



       # print("dev loss is %f", dev_loss)
        print("dev accuracy is %f", dev_accuracy)

        return loss_batches / num_batches

    def fit(self, sess, saver, train, dev):
        losses = []

        for epoch in range(self.config.n_epochs):
            loss = self.run_epoch(sess, saver, train, dev)
            losses.append(loss)

            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))

            print("loss after %d Epoch is %f", epoch, loss)

        return losses

def main(debug=True):

    print(80*"=")
    print("Initializing")
    print(80*"=")

    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building Model......")
        config = Config()
        start = time.time()
        model = MNISTModel(config)
        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        print("Took {:.2f} seconds\n".format(time.time()-start))

    with tf.Session(graph=graph) as session:
        session.run(init)
        mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

        train_set= mnist_data.train
        dev_set = mnist_data.validation
        test_set = mnist_data.test

        losses = model.fit(session,saver,train_set,dev_set)

        test_predictions, test_accuracy = model.predict_on_batch(session, test_set.images, test_set.labels)

        print("test accuracy", test_accuracy)

if __name__ == "__main__":

    main()
    # do_train()

