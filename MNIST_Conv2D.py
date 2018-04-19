import tensorflow as tf
from Model import Model
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.ops import variable_scope as vs

import logging
import os
import time
logger = logging.getLogger("MNIST")
import numpy as np

"""
Main File for MNIST Implementation using Tensorflow 
"""
class Config(object):
    """Holds model hyperparams and data information.
    """
    n_features = 784
    n_classes = 10
    dropout = 0.25
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

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            self.summaries = tf.summary.merge_all()

    def build_graph(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.acc = self.calc_accuracy(self.pred,self.labels_placeholder)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):

        with vs.variable_scope("Input"):
            cn = Config()
            self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,cn.n_features])
            self.labels_placeholder = tf.placeholder(dtype=tf.int32,shape=[None,cn.n_classes])
            self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):

        with vs.variable_scope("Feed_Dict") :

            feed_dict = dict()
            feed_dict[self.input_placeholder]= inputs_batch

            if labels_batch is not None:
                feed_dict[self.labels_placeholder] = labels_batch

            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_prediction_op(self):

        cn = Config()

        # x = tf.reshape(self.input_placeholder,[-1, 28, 28, 1 ])
        # K1=32
        # W1 = tf.get_variable("Weights_Layer1", dtype=tf.float32,shape=(5,5,1,K1), initializer=tf.contrib.layers.xavier_initializer())
        # b1 = tf.get_variable("Bias_Layer1",dtype=tf.float32,shape=[K1],initializer=tf.constant_initializer(0.1))
        #
        # h_conv1 = tf.nn.relu(tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME') + b1 )
        # h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #
        # K2 = 64
        # W2 = tf.get_variable("Weights_Layer2", dtype=tf.float32, shape=(5,5,K1,K2), initializer=tf.contrib.layers.xavier_initializer())
        # b2 = tf.get_variable("Bias_Layer2",dtype=tf.float32,shape=[K2],initializer=tf.constant_initializer(0.1))
        #
        # h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W2,strides=[1,1,1,1],padding='SAME') + b2)
        # h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #
        #
        # K3 = 1024
        # W3 = tf.get_variable("Weights_Layer3", dtype=tf.float32, shape=(7*7*K2, K3),initializer=tf.contrib.layers.xavier_initializer())
        # b3 = tf.get_variable("Bias_Layer3", dtype=tf.float32, shape=[K3], initializer=tf.constant_initializer(0.1))
        #
        # h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W3) + b3)
        #
        # h_fc1  = tf.nn.dropout(h_fc1, 1-self.dropout_placeholder)
        #
        # W4 =  tf.get_variable("Weights_Layer4", dtype=tf.float32, shape=(K3, 10),initializer=tf.contrib.layers.xavier_initializer())
        # b4 = tf.get_variable("Bias_Layer4", dtype=tf.float32, shape=[10], initializer=tf.constant_initializer(0.1))
        #
        # pred = tf.nn.softmax(tf.matmul(h_fc1, W4) + b4)

        input_layer = tf.reshape(self.input_placeholder, [-1, 28, 28, 1])

        #Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 28, 28, 1]
        # Output Tensor Shape: [batch_size, 28, 28, 32]

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)


        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=True)

        pred = tf.layers.dense(inputs=dropout, units=10)

        return pred

    def add_loss_op(self, pred):

        out1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder,logits=pred)
        loss = tf.reduce_mean(out1)

        tf.summary.scalar("loss", loss)
        return loss

    def add_training_op(self, loss):

        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch,summary_Writer):


        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss, accuracy, summaries = sess.run([self.train_op, self.loss, self.acc,self.summaries ], feed_dict=feed)

        return loss,accuracy,summaries

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

    def run_epoch(self, sess, train, summary_writer):

        num_examples = train.images.shape[0]
        batch_size = self.config.batch_size
        num_batches = num_examples / batch_size

        loss_batches = 0
        for i in range(55): #num_batches is float bug here
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_train = train.images[start_idx:end_idx, :]
            y_train = train.labels[start_idx:end_idx, :]

            loss, accuracy,summaries = self.train_on_batch(sess, x_train, y_train, summary_writer)

            loss_batches = loss_batches + loss

            # summary_writer.add_summary(summaries)

        return loss_batches / 55

    def train(self, sess, saver, train, dev):
        """
        Main train loop
        """
        losses = []
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        print("Number of parameters %d"%num_params)

        summary_writer = tf.summary.FileWriter('./Experiments/', sess.graph)

        for epoch in range(self.config.n_epochs):
            loss = self.run_epoch(sess, train, summary_writer)

            losses.append(loss)

            print("Epoch %d out of %d" %(epoch + 1, self.config.n_epochs))

            print("loss after %d Epoch is %f" %(epoch+1, loss))

            # saver.save(sess,'./data/weights')

            self.write_summary(loss,"Train loss",summary_writer, epoch)

            x_dev = dev.images
            y_dev = dev.labels

            pred_dev, dev_accuracy = self.predict_on_batch(sess, x_dev, y_dev)
            # dev_loss = self.loss(pred_dev)
            print("dev accuracy is %f", dev_accuracy)
            # print("dev loss(inside train) is %f", dev_loss)
            # self.write_summary(dev_loss, "Dev Loss", summary_writer, epoch)
            self.write_summary(dev_accuracy, "Dev Accuracy", summary_writer, epoch)

        return losses

    def write_summary(self, value, tag, summary_writer, global_step):
        """Write a single summary value to tensorboard"""
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, global_step)


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

        model.train(session,saver,train_set,dev_set)

        test_predictions, test_accuracy = model.predict_on_batch(session, test_set.images, test_set.labels)

        print("test accuracy", test_accuracy)

if __name__ == "__main__":

    main()
    # do_train()

