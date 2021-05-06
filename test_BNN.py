# TensorFlow and tf.keras
import tensorflow as tf
from src.modeling.models.BNN import BNN
from dotmap import DotMap
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

tf.get_logger().setLevel(logging.ERROR)

model_config = [
    DotMap(
        {
            "layer_name": "FC",
            "input_dim": 784,
            "output_dim": 500,
            "activation": "swish",
            "weight_decay": 0.0001,
            "ensemble_size": 1,
        }
    ),
    # DotMap(
    #     {
    #         "layer_name": "FC",
    #         "input_dim": 500,
    #         "output_dim": 500,
    #         "activation": "swish",
    #         "weight_decay": 0.00025,
    #         "ensemble_size": 1,
    #     }
    # ),
    DotMap(
        {
            "layer_name": "FC",
            "input_dim": 500,
            "output_dim": 500,
            "activation": "swish",
            "weight_decay": 0.00025,
            "ensemble_size": 1,
        }
    ),
    DotMap(
        {
            "layer_name": "FC",
            "input_dim": 500,
            "output_dim": 10,
            "activation": "swish",
            "weight_decay": 0.0005,
            "ensemble_size": 1,
        }
    ),
]
model = BNN(DotMap(name="test", num_networks=1), model_config)
# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Prepare the training dataset.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
epochs = 15
# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    model.trainable = True
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            mean, var = model(x_batch_train)  # Logits for this minibatch
            predictions = model.sample_predictions(mean, var, calibrate=False)
            train_acc_metric.update_state(y_batch_train, predictions)
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, predictions)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    model.trainable = False
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        mean, var = model(x_batch_val)
        predictions = model.sample_predictions(mean, var, calibrate=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, predictions)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))
