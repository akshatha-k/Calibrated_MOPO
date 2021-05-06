from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from src.misc.MBExp import MBExperiment
from src.controllers.MPC import MPC
from src.args import get_args


def main(args):
    # ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    # cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    # cfg.pprint()

    # policy = MPC(cfg.ctrl_cfg, calibrate)
    exp = MBExperiment(args)

    # os.makedirs(exp.logdir)
    # with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
    #     f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":
    args = get_args()
    main(args)