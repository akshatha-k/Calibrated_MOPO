import tensorflow as tf


class RecalibrationLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super(RecalibrationLayer, self).__init__()
        self.out_dim = out_dim
        self.A = self.add_weight(
            name="A",
            shape=[1, self.out_dim],
            initializer="uniform",
            trainable=True,
            dtype=tf.float32,
        )

        self.B = self.add_weight(
            name="B",
            shape=[1, self.out_dim],
            initializer="uniform",
            trainable=True,
            dtype=tf.float32,
        )

    def build(self, input_shape):
        pass

    def get_vars(self):
        return [self.A, self.B]

    def get_output_dim(self):
        return self.out_dim

    @tf.function
    def call(self, x, activation=True):
        out = x * self.A + self.B
        if not activation:
            return out

        return tf.math.sigmoid(out)

    @tf.function
    def inv_call(self, y, activation=True):
        out = y
        if activation:
            out = tf.math.log(y / (1 - y))

        return (out - self.B) / self.A


# if __name__ == "__main__":
#     a = tf.random.uniform(shape=(1, 64))
#     layer = RecalibrationLayer(64)
#     print(layer(a))
