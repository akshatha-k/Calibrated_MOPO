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
    def __init__(self, env_config, args, model_trainer, calibrate=False):
        super().__init__(args)
        self.env_config = env_config
        self.args = args
        self.trainer = model_trainer
        self.obs_cost_fn = self.env_config.obs_cost_fn
        self.ac_cost_fn = self.env_config.ac_cost_fn
        self.dO, self.dU = (
            self.env_config.env.observation_space.shape[0],
            self.env_config.env.action_space.shape[0],
        )
        self.ac_ub, self.ac_lb = (
            self.env_config.env.action_space.high,
            self.env_config.env.action_space.low,
        )
        # self.ac_ub = np.minimum(self.ac_ub, params.ac_ub)
        # self.ac_lb = np.maximum(self.ac_lb, params.ac_lb)
        # self.per = params.per
        self.per = 1  # TODO: see what per does and add an argument
        self.should_calibrate = calibrate

        self.optimizer = optimizers[args.opt_type](
            sol_dim=self.args.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.args.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.args.plan_hor]),
            popsize=self.args.popsize,
            max_iters=self.args.max_iters,
            num_elites=self.args.num_elites,
            alpha=self.args.alpha,
            epsilon=self.args.epsilon,
        )
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.args.plan_hor])
        self.init_var = np.tile(
            np.square(self.ac_ub - self.ac_lb) / 16, [self.args.plan_hor]
        )
        self.train_in = np.array([]).reshape(
            0, self.dU + self.env_config.obs_preproc(np.zeros([1, self.dO])).shape[-1]
        )
        self.train_targs = np.array([]).reshape(
            0,
            self.env_config.targ_proc(
                np.zeros([1, self.dO]), np.zeros([1, self.dO])
            ).shape[-1],
        )
        self.has_been_trained = False
        self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
        # self.ac_seq = tf.placeholder(
        #     shape=[1, self.args.plan_hor * self.dU], dtype=tf.float32
        # )
        # self.pred_cost, self.pred_traj = self.compile_cost(
        #     self.ac_seq, get_pred_trajs=True
        # )

        print(
            "Created an MPC controller, prop mode %s, %d particles. Calibration set to %s"
            % (self.args.prop_type, self.args.npart, self.should_calibrate)
            + ("Ignoring variance." if self.args.ign_var else "")
        )

        # if self.save_all_models:
        #     print(
        #         "Controller will save all models. (Note: This may be memory-intensive."
        #     )
        # if self.log_particles:
        #     print(
        #         "Controller is logging particle predictions (Note: This may be memory-intensive)."
        #     )
        #     self.pred_particles = []
        #     self.pred_costs = []
        # elif self.log_traj_preds:
        #     print("Controller is logging trajectory prediction statistics (mean+var).")
        #     self.pred_means, self.pred_vars = [], []
        #     self.pred_costs = []
        # else:
        #     print("Trajectory prediction logging is disabled.")

    def train(self, obs_trajs, acs_trajs, rews_trajs, logdir=None):
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(
                np.concatenate([self.env_config.obs_preproc(obs[:-1]), acs], axis=-1)
            )
            new_train_targs.append(self.env_config.targ_proc(obs[:-1], obs[1:]))

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

    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.args.plan_hor])
        self.optimizer.reset()

    def act(self, obs, t, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        self.sy_cur_obs = obs

        soln = self.optimizer.obtain_solution(
            self.compile_cost, self.prev_sol, self.init_var, True
        )
        self.prev_sol = np.concatenate(
            [np.copy(soln)[self.per * self.dU :], np.zeros(self.per * self.dU)]
        )
        self.ac_buf = tf.reshape(soln[: self.per * self.dU], [-1, self.dU])
        # self.ac_buf = soln[: self.per * self.dU].reshape(-1, self.dU)

        if get_pred_cost:
            if self.trainer.model.is_tf_model:
                pred_cost, pred_traj = self.compile_cost(soln[None])
            else:
                raise NotImplementedError()
            return self.act(obs, t), pred_cost
        # elif self.log_traj_preds or self.log_particles:
        #     pred_cost, pred_traj = self.trainer.model.sess.run(
        #         [self.pred_cost, self.pred_traj], feed_dict={self.ac_seq: soln[None]}
        #     )
        #     pred_cost, pred_traj = pred_cost[0], pred_traj[:, 0]

        #     self.pred_costs.append(pred_cost)

        #     if self.log_particles:
        #         self.pred_particles.append(pred_traj)
        #     else:
        #         curr_mean = np.mean(pred_traj, axis=1, keepdims=True)
        #         curr_var = np.mean(np.square(pred_traj - curr_mean), axis=1)
        #         self.pred_means.append(
        #             curr_mean.squeeze()
        #         )  # np.mean(pred_traj, axis=1))
        #         self.pred_vars.append(curr_var)

        # if get_pred_cost:
        #     return self.act(obs, t), pred_cost
        return self.act(obs, t)

    def compile_cost(self, ac_seqs, get_pred_trajs=False):
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.args.npart])
        ac_seqs = tf.reshape(ac_seqs, [-1, self.args.plan_hor, self.dU])
        ac_seqs = tf.reshape(
            tf.tile(
                tf.transpose(ac_seqs, [1, 0, 2])[:, :, None], [1, 1, self.args.npart, 1]
            ),
            [self.args.plan_hor, -1, self.dU],
        )
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.args.npart, 1])

        def continue_prediction(t, *args):
            return tf.less(t, self.args.plan_hor)

        if get_pred_trajs:
            pred_trajs = init_obs[None]

            def iteration(t, total_cost, cur_obs, pred_trajs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs),
                    [-1, self.args.npart],
                )
                next_obs = self.env_config.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, pred_trajs

            _, costs, _, pred_trajs = tf.while_loop(
                cond=continue_prediction,
                body=iteration,
                loop_vars=[t, init_costs, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(),
                    init_costs.get_shape(),
                    init_obs.get_shape(),
                    tf.TensorShape([None, None, self.dO]),
                ],
            )

            # Replace nan costs with very high cost
            costs = tf.math.reduce_mean(
                tf.where(tf.math.is_nan(costs), 1e6 * tf.ones_like(costs), costs),
                axis=1,
            )
            pred_trajs = tf.reshape(
                pred_trajs, [self.args.plan_hor + 1, -1, self.args.npart, self.dO]
            )
            return costs, pred_trajs
        else:

            def iteration(t, total_cost, cur_obs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)

                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs),
                    [-1, self.args.npart],
                )
                return (
                    t + 1,
                    total_cost + delta_cost,
                    self.env_config.obs_postproc2(next_obs),
                )

            _, costs, _ = tf.while_loop(
                cond=continue_prediction,
                body=iteration,
                loop_vars=[t, init_costs, init_obs],
            )

            # Replace nan costs with very high cost
            return tf.math.reduce_mean(
                tf.where(tf.math.is_nan(costs), 1e6 * tf.ones_like(costs), costs),
                axis=1,
            )

    def _predict_next_obs(self, obs, acs):
        proc_obs = tf.cast(self.env_config.obs_preproc(obs), tf.float32)
        acs = tf.cast(acs, tf.float32)
        obs = tf.cast(obs, tf.float32)
        if self.trainer.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            if self.args.prop_type == "TS1":
                proc_obs = tf.reshape(
                    proc_obs, [-1, self.args.npart, proc_obs.get_shape()[-1]]
                )
                sort_idxs = tf.math.top_k(
                    tf.random_uniform([tf.shape(proc_obs)[0], self.args.npart]),
                    k=self.args.npart,
                ).indices
                tmp = tf.tile(
                    tf.range(tf.shape(proc_obs)[0])[:, None], [1, self.args.npart]
                )[:, :, None]
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                proc_obs = tf.gather_nd(proc_obs, idxs)
                proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])
            if self.args.prop_type == "TS1" or self.args.prop_type == "TSinf":
                proc_obs, acs = (
                    self._expand_to_ts_format(proc_obs),
                    self._expand_to_ts_format(acs),
                )

            # Obtain model predictions
            inputs = tf.concat([proc_obs, acs], axis=-1)
            mean, var = self.trainer.create_prediction_tensors(inputs)

            if not self.args.ign_var:
                predictions = self.trainer.model.sample_predictions(
                    mean, var, calibrate=self.should_calibrate
                )
                predictions = tf.cast(predictions, tf.float32)
                if self.args.prop_type == "MM":
                    model_out_dim = predictions.get_shape()[-1].value

                    predictions = tf.reshape(
                        predictions, [-1, self.args.npart, model_out_dim]
                    )
                    prediction_mean = tf.math.reduce_mean(
                        predictions, axis=1, keep_dims=True
                    )
                    prediction_var = tf.math.reduce_mean(
                        tf.square(predictions - prediction_mean), axis=1, keep_dims=True
                    )
                    z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
                    samples = prediction_mean + z * tf.math.sqrt(prediction_var)
                    predictions = tf.reshape(samples, [-1, model_out_dim])
            else:
                predictions = mean

            # TS Optimization: Remove additional dimension
            if self.args.prop_type == "TS1" or self.args.prop_type == "TSinf":
                predictions = self._flatten_to_matrix(predictions)
            if self.args.prop_type == "TS1":
                predictions = tf.reshape(
                    predictions, [-1, self.args.npart, predictions.get_shape()[-1]]
                )
                sort_idxs = tf.math.top_k(-sort_idxs, k=self.args.npart).indices
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                predictions = tf.gather_nd(predictions, idxs)
                predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

            return self.env_config.obs_postproc(obs, predictions)
        else:
            raise NotImplementedError()

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(
                    mat,
                    [
                        -1,
                        self.args.ensemble_size,
                        self.args.npart // self.args.ensemble_size,
                        dim,
                    ],
                ),
                [1, 0, 2, 3],
            ),
            [self.args.ensemble_size, -1, dim],
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(
                    ts_fmt_arr,
                    [
                        self.args.ensemble_size,
                        -1,
                        self.args.npart // self.args.ensemble_size,
                        dim,
                    ],
                ),
                [1, 0, 2, 3],
            ),
            [-1, dim],
        )
