from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time, localtime, strftime
from src.modeling.utils.args import get_args

import numpy as np
from scipy.io import savemat
from dotmap import DotMap

from src.misc.DotmapUtils import get_required_argument
from src.misc.Agent import Agent
from src.modeling.trainers.registry import get_trainer

SAVE_EVERY = 25


class MBExperiment:
    def __init__(self, args):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.args = args

        self.env_config = get_config(self.args.env)(self.args)
        self.env = self.env_trainer.env

        self.agent = Agent(self.args, self.env)
        # self.model = self.env_config.nn_constructor()
        self.model_trainer = BNN_trainer(self.args, self.model)
        self.policy = MPC(
            self.env_config, self.args, self.model_trainer
        )  # TODO: Convert MPC and make an object here; we need a get controller here

    def run_experiment(self):
        """Perform experiment."""
        # os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

        # Perform initial rollouts
        samples = []
        for i in range(self.args.ninit_rollouts):
            samples.append(self.agent.sample(self.args.task_hor, self.policy))
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        if self.args.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
            )

        # Training loop
        for i in range(self.args.ntrain_iters):
            print(
                "####################################################################"
            )
            print("Starting training iteration %d." % (i + 1))

            # iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            # os.makedirs(iter_dir, exist_ok=True)

            samples = []
            for j in range(self.args.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor,
                        self.policy,
                        os.path.join(self.args.output_dir, "rollout%d.mp4" % j),
                    )
                )
            # if self.args.nrecord > 0:
            #     for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
            #         os.remove(os.path.join(iter_dir, item))
            for j in range(
                max(self.args.neval, self.args.nrollouts_per_iter) - self.args.nrecord
            ):
                samples.append(self.agent.sample(self.args.task_hor, self.policy))
            print(
                "Rewards obtained:",
                [sample["reward_sum"] for sample in samples[: self.args.neval]],
            )
            traj_obs.extend(
                [sample["obs"] for sample in samples[: self.args.nrollouts_per_iter]]
            )
            traj_acs.extend(
                [sample["ac"] for sample in samples[: self.args.nrollouts_per_iter]]
            )
            traj_rets.extend(
                [sample["reward_sum"] for sample in samples[: self.args.neval]]
            )
            traj_rews.extend(
                [
                    sample["rewards"]
                    for sample in samples[: self.args.nrollouts_per_iter]
                ]
            )
            samples = samples[: self.args.nrollouts_per_iter]

            savemat(
                os.path.join(self.args.output_dir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "rewards": traj_rews,
                },
            )

            if i < self.args.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples],
                )

            # Delete iteration directory if not used
            if len(os.listdir(self.args.output_dir)) == 0:
                os.rmdir(self.args.output_dir)
