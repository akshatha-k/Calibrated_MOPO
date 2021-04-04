from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network."""

    def __init__(self, x_dim):
        """Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.

        Returns: None.
        """
        self.fitted = False
        with tf.name_scope("Scaler"):
            initializer = tf.constant_initializer(0.0)
            self.mu = tf.Variable(
                initial_value=initializer(shape=[1, x_dim]),
                trainable=False,
                name="scaler_mu",
                dtype=tf.float32,
            )
            initializer = tf.constant_initializer(1.0)
            self.sigma = tf.Variable(
                initial_value=initializer(shape=[1, x_dim]),
                trainable=False,
                name="scaler_std",
                dtype=tf.float32,
            )

        self.cached_mu, self.cached_sigma = tf.zeros([0, x_dim]), tf.ones([1, x_dim])

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        mu = tf.math.reduce_mean(data, axis=0, keepdims=True)
        sigma = (tf.math.reduce_std(data, axis=0, keepdims=True)).numpy()
        sigma[sigma < 1e-12] = 1.0
        sigma = tf.cast(sigma, tf.float32)
        # sigma[sigma < 1e-12] = 1.0

        self.mu = tf.cast(mu, tf.float32)
        self.sigma = sigma
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.

        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.

        Returns: None.
        """
        self.cached_mu = self.mu
        self.cached_sigma = self.sigma

    def load_cache(self):
        """Loads values from the cache

        Returns: None.
        """
        self.mu.load(self.cached_mu)
        self.sigma.load(self.cached_sigma)


# if __name__ == "__main__":
#     scaler = TensorStandardScaler(32)
#     print(scaler.get_vars())
