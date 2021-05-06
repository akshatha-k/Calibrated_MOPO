from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from src.args import get_args
from src.modeling.models.BNN import BNN
from src.misc.DotmapUtils import get_required_argument
from src.modeling.layers import FC
from src.modeling.trainers.registry import register
import src.envs


class HalfCheetah:
    MODEL_IN, MODEL_OUT = 24, 18
    GP_NINDUCING_POINTS = 300

    def __init__(self, args):
        self.args = args
        if args.mujoco:
            self.env = gym.make("MBRLHalfCheetah-v0")

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate(
                [obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]],
                axis=1,
            )
        else:
            return tf.concat(
                [obs[:, 1:2], tf.sin(obs[:, 2:3]), tf.cos(obs[:, 2:3]), obs[:, 3:]],
                axis=1,
            )

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)
        else:
            return tf.concat([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)

    @staticmethod
    def obs_postproc2(next_obs):
        return next_obs

    @staticmethod
    def targ_proc(obs, next_obs):
        return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)
        else:
            return 0.1 * tf.math.reduce_sum(tf.math.square(acs), axis=1)

    def nn_constructor(self):
        if not self.args.load_model:
            model_config = [
                DotMap(
                    {
                        "layer_name": "FC",
                        "input_dim": self.MODEL_IN,
                        "output_dim": 200,
                        "activation": "swish",
                        "weight_decay": 0.00005,
                        "ensemble_size": 1,
                    }
                ),
                DotMap(
                    {
                        "layer_name": "FC",
                        "input_dim": 200,
                        "output_dim": 200,
                        "activation": "swish",
                        "weight_decay": 0.000075,
                        "ensemble_size": 1,
                    }
                ),
                DotMap(
                    {
                        "layer_name": "FC",
                        "input_dim": 200,
                        "output_dim": 200,
                        "activation": "swish",
                        "weight_decay": 0.000075,
                        "ensemble_size": 1,
                    }
                ),
                DotMap(
                    {
                        "layer_name": "FC",
                        "input_dim": 200,
                        "output_dim": self.MODEL_OUT,
                        "activation": "swish",
                        "weight_decay": 0.0001,
                        "ensemble_size": 1,
                    }
                ),
            ]
            model = BNN(DotMap(name="test", num_networks=1), model_config)
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(
            model_init_cfg, "model_class", "Must provide model class"
        )(
            DotMap(
                name="model",
                kernel_class=get_required_argument(
                    model_init_cfg, "kernel_class", "Must provide kernel class"
                ),
                kernel_args=model_init_cfg.get("kernel_args", {}),
                num_inducing_points=get_required_argument(
                    model_init_cfg,
                    "num_inducing_points",
                    "Must provide number of inducing points.",
                ),
                sess=self.SESS,
            )
        )
        return model


@register
def halfcheetah(args):
    return HalfCheetah(args)