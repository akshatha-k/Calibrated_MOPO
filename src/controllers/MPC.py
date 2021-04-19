import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from src.modeling.trainers.BNN_trainer import BNN_trainer
from src.misc.DotmapUtils import get_required_argument
from src.misc.optimizers import RandomOptimizer, CEMOptimizer

from sklearn.model_selection import train_test_split

optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}


class MPC(Controller):
    def __init__(self, env, params, calibrate=False):
        super().__init__(params)
        self.dO, self.dU = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.ac_ub, self.ac_lb = (
            env.action_space.high,
            env.action_space.low,
        )
        self.ac_ub = np.minimum(self.ac_ub, params.ac_ub)
        self.ac_lb = np.maximum(self.ac_lb, params.ac_lb)
        self.per = params.per
        self.should_calibrate = calibrate

        self.model = None  # TODO: initialize the model here
        self.model_trainer = BNN_trainer(model_params, self.model)
        self.optimizer = optimizers[params.optim](
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            **opt_cfg
        )
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(
            np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor]
        )
        self.train_in = np.array([]).reshape(
            0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1]
        )
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )

        self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
        self.ac_seq = tf.placeholder(
            shape=[1, self.plan_hor * self.dU], dtype=tf.float32
        )
        self.pred_cost, self.pred_traj = self._compile_cost(
            self.ac_seq, get_pred_trajs=True
        )

        print(
            "Created an MPC controller, prop mode %s, %d particles. Calibration set to %s"
            % (self.prop_mode, self.npart, self.should_calibrate)
            + ("Ignoring variance." if self.ign_var else "")
        )

        if self.save_all_models:
            print(
                "Controller will save all models. (Note: This may be memory-intensive."
            )
        if self.log_particles:
            print(
                "Controller is logging particle predictions (Note: This may be memory-intensive)."
            )
            self.pred_particles = []
            self.pred_costs = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
            self.pred_costs = []
        else:
            print("Trajectory prediction logging is disabled.")

    def train(self, obs_trajs, acs_trajs, rews_trajs, logdir=None):
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(
                np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1)
            )
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))

        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        train_in, cal_in, train_targs, cal_targs = train_test_split(
            self.train_in, self.train_targs, test_size=0.2
        )

        # Train the model
        self.trainer.train(train_in, train_targs)
        # calibrate the model
        if self.should_calibrate:
            self.trainer.calibrate(cal_in, cal_targs)

        self.has_been_trained = True