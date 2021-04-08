from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import trange
from scipy.io import savemat, loadmat
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

from datetime import datetime

from src.modeling.utils.TensorStandardScaler import TensorStandardScaler
from src.modeling.layers.FC_v2 import FC
from src.modeling.layers.RecalibrationLayer import RecalibrationLayer
from src.misc.DotmapUtils import *

import math


class BNN_trainer:
    def __init__(self, params, model):
        self.params = params
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.num_nets = params.num_nets

        self.model = model
        self.model_dir = params.model_dir

        # Training objects
        self.optimizer = tf.keras.optimizers.Adam()
        self.mse_loss = None

        # Prediction objects
        self.sy_pred_mean, self.sy_pred_var = (
            None,
            None,
        )

        self.cal_optimizer = tf.keras.optimizers.Adam()
        self.cal_loss = None
        # TODO: saving and loading model

    @tf.function
    def compute_losses(self, targets, mean, log_var, incl_var_loss=True):
        inv_var = tf.math.exp(-log_var)
        if incl_var_loss:
            mse_losses = tf.math.reduce_mean(
                tf.math.reduce_mean(tf.math.square(mean - targets) * inv_var, axis=-1),
                axis=-1,
            )
            var_losses = tf.math.reduce_mean(
                tf.math.reduce_mean(log_var, axis=-1), axis=-1
            )
            total_losses = mse_losses + var_losses
        else:
            total_losses = tf.math.reduce_mean(
                tf.reduce_mean(tf.math.square(mean - targets), axis=-1), axis=-1
            )

        return total_losses

    @tf.function
    def train_step(self, inputs, targets):
        inputs = tf.cast(inputs, dtype=tf.float32)
        targets = tf.cast(targets, dtype=tf.float32)
        with tf.name_scope("train_step"):
            with tf.GradientTape() as tape:
                mean, log_var = self.model(inputs, ret_log_var=True)
                train_loss = tf.math.reduce_sum(
                    self.compute_losses(targets, mean, log_var, True)
                )
                # train_loss+= #TODO: Add Decays to the Loss Function
                train_loss += 0.01 * tf.math.reduce_sum(
                    self.model.max_logvar
                ) - 0.01 * tf.math.reduce_sum(self.model.min_logvar)
            grads = tape.gradient(train_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                grads_and_vars=zip(grads, self.model.trainable_variables),
                name="gradient_application_train_step",
            )
            mse_loss = self.compute_losses(targets, mean, log_var, False)
        return train_loss, mse_loss

    # TODO: epochs and batch_size
    def train(
        self, inputs, targets, hide_progress=False, holdout_ratio=0.2, max_logging=1000
    ):
        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = (
            inputs[permutation[num_holdout:]],
            inputs[permutation[:num_holdout]],
        )
        targets, holdout_targets = (
            targets[permutation[num_holdout:]],
            targets[permutation[:num_holdout]],
        )
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])
        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])

        self.model.scaler.fit(inputs)

        if hide_progress:
            epoch_range = range(self.epochs)
        else:
            epoch_range = trange(self.epochs, unit="epoch(s)", desc="Network training")

        for epoch in epoch_range:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / self.batch_size))):
                batch_idxs = idxs[
                    :, batch_num * self.batch_size : (batch_num + 1) * self.batch_size
                ]
                # Call train step
                train_loss, mse_loss = self.train_step(
                    inputs[batch_idxs], targets[batch_idxs]
                )
            idxs = shuffle_rows(idxs)
            # TODO: holdout loss
            if not hide_progress:
                if holdout_ratio < 1e-12:
                    epoch_range.set_postfix({"Training loss(es)": mse_loss})
                else:
                    epoch_range.set_postfix(
                        {"Training loss(es)": mse_loss, "Holdout loss(es)": mse_loss}
                    )

    def create_prediction_tensors(self, inputs, factored=False):
        factored_mean, factored_variance = self.model(inputs)
        if len(inputs.shape) == 2 and not factored:
            mean = tf.math.reduce_mean(factored_mean, axis=0)
            variance = tf.math.reduce_mean(
                tf.math.square(factored_mean - mean), axis=0
            ) + tf.math.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

    def predict(self, inputs, factored=False):
        with tf.name_scope("create_predict_tensors"):
            self.sy_pred_mean, self.sy_pred_var = self.create_prediction_tensors(
                inputs, factored
            )

    @tf.function
    def cal_step(self, inputs, targets):
        with tf.name_scope("cal_step"):
            with tf.GradientTape() as tape:
                cdf_pred = self.model.recalibrator(inputs)
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets, logits=cdf_pred
                )
                self.cal_loss = tf.math.reduce_mean(
                    tf.math.reduce_mean(cross_entropy, axis=-1), axis=-1
                )
            grads = tape.gradient(self.cal_loss, self.model.cal_vars)
            self.cal_optimizer.apply_gradients(
                grads_and_vars=zip(grads, self.model.cal_vars), name="cal_step"
            )
        return self.cal_loss

    def calibrate(
        self, inputs, targets, hide_progress=False, holdout_ratio=0.0, max_logging=5000
    ):
        inputs, targets = tf.cast(inputs, dtype=tf.float32), tf.cast(
            targets, dtype=tf.float32
        )
        self.model.scaler.fit(inputs)
        self.predict(inputs)
        all_ys = targets

        train_x = np.zeros_like(all_ys)
        train_y = np.zeros_like(all_ys)

        for d in range(self.sy_pred_mean.shape[1]):
            mu = self.sy_pred_mean[:, d]
            var = self.sy_pred_var[:, d]
            ys = all_ys[:, d]

            cdf_pred = norm.cdf(ys, loc=mu, scale=tf.math.sqrt(var))
            cdf_true = np.array(
                [np.sum(cdf_pred < p) / len(cdf_pred) for p in cdf_pred]
            )

            train_x[:, d] = cdf_pred
            train_y[:, d] = cdf_true

        if hide_progress:
            epoch_range = range(self.epochs)
        else:
            epoch_range = trange(
                self.epochs, unit="epoch(s)", desc="Calibration training"
            )

        def iterate_minibatches(inp, targs, batchsize, shuffle=True):
            assert inp.shape[0] == targs.shape[0]
            indices = np.arange(inp.shape[0])
            if shuffle:
                np.random.shuffle(indices)

            last_idx = 0

            for curr_idx in range(
                0, inp.shape[0] - self.batch_size + 1, self.batch_size
            ):
                curr_batch = indices[curr_idx : curr_idx + self.batch_size]
                last_idx = curr_idx + self.batch_size
                yield inp[curr_batch], targs[curr_batch]

            if inp.shape[0] % self.batch_size != 0:
                last_batch = indices[last_idx:]
                yield inp[last_batch], targs[last_batch]

        for _ in epoch_range:
            for x_batch, y_batch in iterate_minibatches(
                train_x, train_y, self.batch_size
            ):
                self.cal_loss = self.cal_step(x_batch, y_batch)

            if not hide_progress:
                epoch_range.set_postfix({"Training loss(es)": self.cal_loss})


if __name__ == "__main__":
    from dotmap import DotMap

    NUM_SAMPLES = 1024
    IN_DIM = 100
    HIDDEN_DIM = 10
    OUT_DIM = 2

    model_config = [
        DotMap(
            {
                "layer_name": "FC",
                "input_dim": 32,
                "output_dim": 32,
                "activation": "swish",
                "weight_decay": 0.05,
                "ensemble_size": 1,
            }
        ),
        DotMap(
            {
                "layer_name": "FC",
                "input_dim": 32,
                "output_dim": 4,
                "activation": "swish",
                "weight_decay": 0.05,
                "ensemble_size": 1,
            }
        ),
    ]
    model = BNN(DotMap(name="test"), model_config)
    a = tf.random.uniform(shape=(32, 32))
    print(model(a)[0])