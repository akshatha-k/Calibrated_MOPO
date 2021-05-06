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


class BNN(tf.keras.Model):
    """Neural network models which model aleatoric uncertainty (and possibly epistemic uncertainty
    with ensembling).
    """

    def __init__(self, params, model_config):
        super(BNN, self).__init__()
        self.num_nets = params.get("num_networks")
        self.model_config = model_config
        self.is_tf_model = True
        self.total_modules = len(self.model_config)
        self.modules = []
        self.decays = []
        self.cal_vars = None
        self.end_act, self.end_act_name = None, None
        self.build_model()
        self.model_dir = params.get("model_dir", None)
        self.scaler = TensorStandardScaler(self.modules[0].get_input_dim())
        self.max_logvar = tf.Variable(
            np.ones([1, self.modules[-1].get_output_dim() // 2]) / 2.0,
            dtype=tf.float32,
            name="max_log_var",
        )
        self.min_logvar = tf.Variable(
            -np.ones([1, self.modules[-1].get_output_dim() // 2]) * 10.0,
            dtype=tf.float32,
            name="min_log_var",
        )

    def build_model(self):
        for i, layer_config in enumerate(self.model_config):
            if layer_config.layer_name == "FC":
                output_dim = layer_config.output_dim
                activation = layer_config.activation
                if i == (self.total_modules - 1):
                    output_dim = (
                        2 * output_dim
                    )  # Returns the mean and variance corresponding to each of the actions
                    activation = None
                layer = FC(
                    output_dim=output_dim,
                    input_dim=layer_config.input_dim,
                    activation=activation,
                    weight_decay=layer_config.weight_decay,
                    ensemble_size=layer_config.ensemble_size,
                )
            self.modules.append(layer)
            self.decays.append(layer_config.weight_decay)
        self.recalibrator = RecalibrationLayer(self.model_config[-1].output_dim)
        self.end_act = self.modules[-1].get_activation()
        self.end_act_name = self.modules[-1].get_activation(as_func=False)
        self.cal_vars = self.recalibrator.get_vars()

    def call(self, inputs, ret_log_var=False):
        """Compiles the output of the network at the given inputs.
        If inputs is 2D, returns a 3D tensor where output[i] is the output of the ith network in the ensemble.
         If inputs is 3D, returns a 3D tensor where output[i] is the output of the ith network on the ith input matrix.
         Arguments:
             inputs: (tf.Tensor) A tensor representing the inputs to the network
             ret_log_var: (bool) If True, returns the log variance instead of the variance.
         Returns: (tf.Tensors) The mean and variance/log variance predictions at inputs for each network
             in the ensemble.
        """
        dim_output = self.modules[-1].get_output_dim()
        cur_out = self.scaler.transform(inputs)
        for layer in self.modules:
            cur_out = layer(cur_out)

        mean = cur_out[:, :, : dim_output // 2]
        if self.end_act is not None:
            mean = self.end_act(mean)

        logvar = self.max_logvar - tf.math.softplus(
            self.max_logvar - cur_out[:, :, dim_output // 2 :]
        )
        logvar = self.min_logvar + tf.math.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        else:
            return mean, tf.math.exp(logvar)

    def sample_predictions(self, means, var, calibrate=True):
        """
        Input shape of mean and var is N x d where N
        is batch size and d is size of state space dimension
        """
        if not calibrate:
            outputs = means + tf.random.normal(
                shape=tf.shape(means), mean=0, stddev=1
            ) * tf.sqrt(var)
            outputs = tf.math.softmax(tf.math.reduce_mean(outputs, axis=0))
            return outputs

        ps = tf.random.uniform(shape=means.shape)

        ps = self.recalibrator.inv_call(ps, activation=True)
        ps = tf.clip_by_value(ps, 1e-6, 1 - 1e-6)

        dist = tfp.distributions.Normal(loc=means, scale=tf.sqrt(var))
        ret = dist.quantile(ps)
        ret = tf.math.reduce_mean(ret, axis=0)
        return tf.math.softmax(ret)