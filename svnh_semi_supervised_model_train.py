import pickle as pkl
import time
import os

import numpy as np
import tensorflow as tf
from dataset import Dataset
from gan import GAN
import utils


checkpoints_dir = 'checkpoints/'

def common_init():
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    y = tf.placeholder(tf.int32, (None), name='y')
    label_mask = tf.placeholder(tf.int32, (None), name='label_mask')

    return inputs_real, inputs_z, y, label_mask


def train(net, dataset, epochs, batch_size, z_size):
    saver = tf.train.Saver()
    sample_z = np.random.normal(0, 1, size=(50, z_size))

    samples, train_accuracies, test_accuracies = [], [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            print("Epoch", e)

            t1e = time.time()
            num_examples = 0
            num_correct = 0
            for x, y, label_mask in dataset.batches(batch_size, dataset, "train"):
                assert 'int' in str(y.dtype)
                steps += 1
                num_examples += label_mask.sum()

                # Sample random noise for G
                batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                # Run optimizers
                t1 = time.time()
                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: x, net.input_z: batch_z,
                                                    net.y : y, net.label_mask : label_mask})
                t2 = time.time()
                num_correct += correct

            sess.run([net.shrink_lr])

            train_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier train accuracy: ", train_accuracy)

            num_examples = 0
            num_correct = 0
            for x, y in dataset.batches(batch_size, dataset, "test"):
                assert 'int' in str(y.dtype)
                num_examples += x.shape[0]

                correct, = sess.run([net.correct], feed_dict={
                    net.input_real: x, net.y : y, net.drop_rate: 0.})
                num_correct += correct

            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy", test_accuracy)
            print("\t\tStep time: ", t2 - t1)
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)

            gen_samples = sess.run(net.samples, feed_dict={
                net.input_z: sample_z})
            samples.append(gen_samples)

            # Save history of accuracies to view after training
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        saver.save(sess, './checkpoints/generator.ckpt')

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return train_accuracies, test_accuracies, samples


def main():
    common_init()
    utils.download_train_and_test_data()
    trainset, testset = utils.load_data_sets()

    real_size = (32, 32, 3)
    z_size = 100
    learning_rate = 0.0003

    tf.reset_default_graph()
    input_real = tf.placeholder(tf.float32, (None, *real_size), name='input_real')
    net = GAN(input_real, z_size, learning_rate)

    dataset = Dataset(trainset, testset)

    batch_size = 128
    epochs = 25
    _, _, _ = train(net, dataset, epochs, batch_size, z_size)


if __name__ == '__main__':
    main()
