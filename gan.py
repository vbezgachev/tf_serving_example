import tensorflow as tf

class GAN:
    """
    Creates and stores inputs, outputs, generator and discriminator of the GAN model
    """
    def __init__(self, input_real, z_size, learning_rate, num_classes=10,
                 alpha=0.2, beta1=0.5, drop_rate=.5):
        """
        Initializes the GAN model.

        :param input_real: Real data for the discriminator
        :param z_size: The number of entries in the noise vector.
        :param learning_rate: The learning rate to use for Adam optimizer.
        :param num_classes: The number of classes to recognize.
        :param alpha: The slope of the left half of the leaky ReLU activation
        :param beta1: The beta1 parameter for Adam.
        :param drop_rate: RThe probability of dropping a hidden unit (used in discriminator)
        """

        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.input_real = input_real
        self.input_z = tf.placeholder(tf.float32, (None, z_size), name='input_z')
        self.y = tf.placeholder(tf.int32, (None), name='y')
        self.label_mask = tf.placeholder(tf.int32, (None), name='label_mask')
        self.drop_rate = tf.placeholder_with_default(drop_rate, (), "drop_rate")

        loss_results = self.model_loss(self.input_real, self.input_z,
                                       self.input_real.shape[3], self.y, num_classes,
                                       label_mask=self.label_mask,
                                       drop_rate=self.drop_rate,
                                       alpha=alpha)

        self.d_loss, self.g_loss, self.correct, \
            self.masked_correct, self.samples, self.pred_class, \
                self.discriminator_class_logits, self.discriminator_out = \
                    loss_results

        self.d_opt, self.g_opt, self.shrink_lr = self.model_opt(self.d_loss,
                                                                self.g_loss,
                                                                self.learning_rate, beta1)


    def model_loss(self, input_real, input_z, output_dim, y, num_classes,
                   label_mask, drop_rate, alpha=0.2):
        """
        Get the loss for the discriminator and generator

        :param input_real: Images from the real dataset
        :param input_z: Noise input of the generator
        :param output_dim: The number of channels in the output image
        :param y: Integer class labels
        :param num_classes: The number of classes to recognize
        :param label_mask: Masks the labels that should be ignored by the semi-suprvised learning
        :param drop_rate: The probability of dropping a hidden unit (used in discriminator)
        :param alpha: The slope of the left half of leaky ReLU activation
        :return: A tuple of (discriminator loss, generator loss)
        """

        # These numbers multiply the size of each layer of the generator and the discriminator,
        # respectively. You can reduce them to run your code faster for debugging purposes.
        g_size_mult = 32
        d_size_mult = 64

        # Here we run the generator and the discriminator
        g_model = self.generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)
        d_on_data = self.discriminator(input_real, drop_rate=drop_rate, alpha=alpha,
                                       size_mult=d_size_mult)
        out, class_logits_on_data, gan_logits_on_data, data_features = d_on_data
        d_on_samples = self.discriminator(g_model, drop_rate=drop_rate, reuse=True, alpha=alpha,
                                          size_mult=d_size_mult)
        _, _, gan_logits_on_samples, sample_features = d_on_samples

        # Here we compute `d_loss`, the loss for the discriminator.
        # This should combine two different losses:
        #  1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
        #     real-vs-fake classification problem.
        #  2. The loss for the SVHN digit classification problem, where we minimize the
        #     cross-entropy for the multi-class softmax. For this one we use the labels.
        #     Don't forget to ignore use `label_mask` to ignore the examples that we
        #     are pretending are unlabeled for the semi-supervised learning problem.
        d_gan_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=gan_logits_on_data, labels=tf.ones_like(gan_logits_on_data)))

        d_gan_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=gan_logits_on_samples, labels=tf.zeros_like(gan_logits_on_samples)))

        d_gan_loss = d_gan_loss_real + d_gan_loss_fake

        y = tf.squeeze(y)
        d_class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=class_logits_on_data, labels=tf.one_hot(y, num_classes, dtype=tf.float32))
        d_class_cross_entropy = tf.squeeze(d_class_cross_entropy)

        label_mask = tf.squeeze(tf.to_float(label_mask))
        d_class_loss = tf.reduce_sum(label_mask*d_class_cross_entropy) / tf.maximum(
            1., tf.reduce_sum(label_mask))
        d_loss = d_gan_loss + d_class_loss

        # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
        # This loss consists of minimizing the absolute difference between the expected features
        # on the data and the expected features on the generated samples.
        # This loss works better for semi-supervised learning than the tradition GAN losses.
        data_features_mean = tf.reduce_mean(data_features, axis=0)
        sample_features_mean = tf.reduce_mean(sample_features, axis=0)
        g_loss = tf.reduce_mean(tf.abs(data_features_mean - sample_features_mean))

        pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32, name='pred_class')
        eq = tf.equal(y, pred_class)
        correct = tf.reduce_sum(tf.to_float(eq), name='correct_pred_sum')
        masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))

        return d_loss, g_loss, correct, masked_correct, g_model, pred_class,\
            class_logits_on_data, out


    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations

        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning rate placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tripple of (discriminator training operation, generator training operation,
                 shrink learning rate)
        """
        # Get weights and biases to update. Get them separately for the discriminator and
        # the generator
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        for t in t_vars:
            assert t in d_vars or t in g_vars

        # Minimize both players' costs simultaneously
        d_train_opt = tf.train. \
                        AdamOptimizer(learning_rate=learning_rate, beta1=beta1). \
                            minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train. \
                AdamOptimizer(learning_rate=learning_rate, beta1=beta1). \
                    minimize(g_loss, var_list=g_vars)

        shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

        return d_train_opt, g_train_opt, shrink_lr


    def generator(self, z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
        '''
        Create a generator for the GAN model

        :param z: Noise input
        :param output_dim: The number of channels in the output image
        :param reuse: Whether the variables should be reused in the generator scope
        :param alpha: The slope of the left half of leaky ReLU activation
        :param training: Whether we are in training mode. Using of the batch normalization depends
                         on this parammeter
        :param size_mult: Multiplication size of each layer of the generator
        '''
        with tf.variable_scope('generator', reuse=reuse):
            # First fully connected layer
            x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
            # Reshape it to start the convolutional stack
            x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))
            x1 = tf.layers.batch_normalization(x1, training=training)
            x1 = tf.maximum(alpha * x1, x1)

            x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
            x2 = tf.layers.batch_normalization(x2, training=training)
            x2 = tf.maximum(alpha * x2, x2)

            x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
            x3 = tf.layers.batch_normalization(x3, training=training)
            x3 = tf.maximum(alpha * x3, x3)

            # Output layer
            logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')
            out = tf.tanh(logits)

            return out


    def discriminator(self, x, drop_rate, reuse=False, alpha=0.2, num_classes=10, size_mult=64):
        '''
        Create a dicriminator for the GAN model

        :param x: Input image (real or fake)
        :param drop_rate: The probability of dropping a hidden unit
        :param reuse: Whether the variables should be reused in the generator scope
        :param alpha: The slope of the left half of leaky ReLU activation
        :param num_classes: The number of classes to recognize
        :param size_mult: Multiplication size of each layer of the generator
        '''

        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.dropout(x, rate=drop_rate/2.5)

            # Input layer is 32x32x3
            x1 = tf.layers.conv2d(x, size_mult, 3, strides=2, padding='same')
            relu1 = tf.maximum(alpha * x1, x1)
            relu1 = tf.layers.dropout(relu1, rate=drop_rate)

            x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
            bn2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = tf.maximum(alpha * bn2, bn2)

            x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same')
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(alpha * bn3, bn3)
            relu3 = tf.layers.dropout(relu3, rate=drop_rate)

            x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same')
            bn4 = tf.layers.batch_normalization(x4, training=True)
            relu4 = tf.maximum(alpha * bn4, bn4)

            x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same')
            bn5 = tf.layers.batch_normalization(x5, training=True)
            relu5 = tf.maximum(alpha * bn5, bn5)

            x6 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=2, padding='same')
            bn6 = tf.layers.batch_normalization(x6, training=True)
            relu6 = tf.maximum(alpha * bn6, bn6)
            relu6 = tf.layers.dropout(relu6, rate=drop_rate)

            x7 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=1, padding='valid')
            # Don't use bn on this layer, because bn would set the mean of each feature
            # to the bn mu parameter.
            # This layer is used for the feature matching loss, which only works if
            # the means can be different when the discriminator is run on the data than
            # when the discriminator is run on the generator samples.
            relu7 = tf.maximum(alpha * x7, x7)

            # Flatten it by global average pooling
            features = tf.reduce_mean(relu7, [1, 2])

            # Set class_logits to be the inputs to a softmax distribution over the different classes
            class_logits = tf.layers.dense(
                features,
                units=num_classes,
                name='discriminator_class_logits')

            # Set gan_logits such that P(input is real | input) = sigmoid(gan_logits).
            # Keep in mind that class_logits gives you the probability distribution over
            # all the real classes and the fake class. You need to work out how to
            # transform this multiclass softmax distribution into a binary real-vs-fake
            # decision that can be described with a sigmoid.
            # Numerical stability is very important.
            # You'll probably need to use this numerical stability trick:
            # log sum_i exp a_i = m + log sum_i exp(a_i - m).
            # This is numerically stable when m = max_i a_i.
            # (It helps to think about what goes wrong when...
            #   1. One value of a_i is very large
            #   2. All the values of a_i are very negative
            # This trick and this value of m fix both those cases, but the naive implementation and
            # other values of m encounter various problems)

            max_val = tf.reduce_max(class_logits, 1, keep_dims=True)
            stable_class_logits = class_logits - max_val
            max_val = tf.squeeze(max_val)
            gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_class_logits), 1)) + max_val

            out = tf.nn.softmax(class_logits, name='discriminator_out')

            return out, class_logits, gan_logits, features
