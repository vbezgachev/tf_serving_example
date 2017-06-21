import pickle as pkl
import time
import os

import numpy as np
import tensorflow as tf
from dataset import Dataset
from gan import GAN
import utils


'''
Trains the GAN model
'''

# constants
checkpoints_dir = 'checkpoints/'

def create_checkpoints_dir():
    '''
    Creates the checkpoints directory if it does not exist
    '''
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


def train(net, dataset, epochs, batch_size, z_size):
    '''
    Main train loop
    :param net: GAN model
    :param dataset: Dataset of images and batches
    :param epochs: Number of train epochs
    :param batch_size: Image batch size
    :param z_size: Size for the noise vector
    :return: Tripple of (train accuracies, test accuracies, generated samples)
    '''
    saver = tf.train.Saver()

    # noise to generate the fake images; it used used at the end
    # of each epoch to check how good the generator is
    sample_z = np.random.normal(0, 1, size=(50, z_size))

    # helpers
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

                # sample random noise for G
                batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                # run optimizers
                t1 = time.time()
                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: x, net.input_z: batch_z,
                                                    net.y : y, net.label_mask : label_mask})
                t2 = time.time()
                num_correct += correct

            # run learning rate adjustment
            sess.run([net.shrink_lr])

            # calcualte and print train statistic
            train_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier train accuracy: ", train_accuracy)

            # run prediction on test images
            num_examples = 0
            num_correct = 0
            for x, y in dataset.batches(batch_size, dataset, "test"):
                assert 'int' in str(y.dtype)
                num_examples += x.shape[0]

                correct, = sess.run([net.correct], feed_dict={
                    net.input_real: x, net.y : y, net.drop_rate: 0.})
                num_correct += correct

            # calculate and print test statistic
            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy", test_accuracy)
            print("\t\tStep time: ", t2 - t1)
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)

            # generate samples for visual check
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
    # preparations
    create_checkpoints_dir()
    utils.download_train_and_test_data()
    trainset, testset = utils.load_data_sets()

    # create real input for the GAN model (its dicriminator) and
    # GAN model itself
    real_size = (32, 32, 3)
    z_size = 100
    learning_rate = 0.0003

    tf.reset_default_graph()
    input_real = tf.placeholder(tf.float32, (None, *real_size), name='input_real')
    net = GAN(input_real, z_size, learning_rate)

    # craete dataset
    dataset = Dataset(trainset, testset)

    # train the model
    batch_size = 128
    epochs = 25
    _, _, _ = train(net, dataset, epochs, batch_size, z_size)


if __name__ == '__main__':
    main()
