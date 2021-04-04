from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

activations = {
    None: tf.identity,
    "ReLU": tf.nn.relu,
    "tanh": tf.math.tanh,
    "sigmoid": tf.math.sigmoid,
    "softmax": tf.nn.softmax,
    "swish": lambda x: x * tf.math.sigmoid(x),
}


class FC(tf.keras.layers.Layer):
    """Represents a fully-connected layer in a network."""

    def __init__(
        self,
        output_dim,
        input_dim=None,
        activation=None,
        weight_decay=None,
        ensemble_size=1,
    ):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weight_decay = weight_decay
        self.ensemble_size = ensemble_size

    def build(self, input_shape):
        initializer = tf.keras.initializers.truncated_normal(
            stddev=1 / (2 * tf.sqrt(tf.cast(self.input_dim, dtype=tf.float32)))
        )
        self.kernel = tf.Variable(
            initial_value=initializer(
                shape=[self.ensemble_size, self.input_dim, self.output_dim]
            ),
            trainable=True,
            name="FC_weights",
            dtype=tf.float32,
        )
        initializer = tf.constant_initializer(0.0)
        self.biases = tf.Variable(
            initial_value=initializer(shape=[self.ensemble_size, 1, self.output_dim]),
            trainable=True,
            name="FC_biases",
            dtype=tf.float32,
        )
        if self.weight_decay is not None:
            self.decays = [
                tf.math.multiply(
                    self.weight_decay, tf.nn.l2_loss(self.kernel), name="weight_decay"
                )
            ]

    def get_input_dim(self):
        """Returns the dimension of the input.

        Returns: The dimension of the input
        """
        return self.input_dim

    def get_output_dim(self):
        """Returns the dimension of the output.

        Returns: The dimension of the output.
        """
        return self.output_dim

    def get_activation(self, as_func=True):
        """Returns the current activation function for this layer.

        Arguments:
            as_func: (bool) Determines whether the returned value is the string corresponding
                     to the activation function or the activation function itself.

        Returns: The activation function (string/function, see as_func argument for details).
        """
        if as_func:
            return activations[self.activation]
        else:
            return self.activation

    def get_ensemble_size(self):
        return self.ensemble_size

    @tf.function
    def call(self, input_tensor):
        """Returns the resulting tensor when all operations of this layer are applied to input_tensor.

        If input_tensor is 2D, this method returns a 3D tensor representing the output of each
        layer in the ensemble on the input_tensor. Otherwise, if the input_tensor is 3D, the output
        is also 3D, where output[i] = layer_ensemble[i](input[i]).

        Arguments:
            input_tensor: (tf.Tensor) The input to the layer.

        Returns: The output of the layer, as described above.
        """
        # Get raw layer outputs
        if len(input_tensor.shape) == 2:
            raw_output = (
                tf.einsum("ij,ajk->aik", input_tensor, self.kernel) + self.biases
            )
        elif (
            len(input_tensor.shape) == 3 and input_tensor.shape[0] == self.ensemble_size
        ):
            raw_output = tf.linalg.matmul(input_tensor, self.kernel) + self.biases
        else:
            raise ValueError("Invalid input dimension.")

        # Apply activations if necessary
        return activations[self.activation](raw_output)


# if __name__ == "__main__":
#     a = tf.random.uniform(shape=(32, 32))
#     layer = FC(64, 32)
#     print(layer(a))
