import argparse

parser = argparse.ArgumentParser(description="Calibrated MOPO")
parser.add_argument("--output_dir", default="/home/projects/mopo/log", type=str)
# Model Parameters
parser.add_argument("--load_model", default=False, type=bool)
parser.add_argument("--ensemble_size", default=1, type=int, help="Ensemble size to use")
parser.add_argument(
    "--model_type", default="P", type=str, choices=["P", "PE", "D", "DE"]
)
parser.add_argument("--epochs", default=50, type=int)  # check if epochs is ntrain_iters
# Trainer Parameters
parser.add_argument("--batch_size", default=1, type=int, help="batch_size to use")

# Experiment parameters
parser.add_argument("--env", default="cartpole", type=str)
# Simulator Parameters
parser.add_argument("--task_hor", default=200, type=int)
parser.add_argument("--stochastic", default=False, type=bool)
parser.add_argument("--noise_std", default=0.0, type=float)
parser.add_argument("--noisy_actions", default=False, type=bool)
# Exp parameters
parser.add_argument("--ntrain_iters", default=50, type=int)
parser.add_argument("--nrollouts_per_iter", default=1, type=int)
parser.add_argument("--ninit_rollouts", default=10, type=int)

# Controller parameters
parser.add_argument("--npart", default=1, type=int)
parser.add_argument("--ign_var", default=1, type=int)
parser.add_argument("--plan_hor", default=25, type=int)
parser.add_argument(
    "--prop_type", default="TSinf", type=str, choices=["E", "DS", "TSinf", "TS1", "MM"]
)
# Optimizer parameters
parser.add_argument("--opt_type", default="CEM", type=str)
parser.add_argument("--max_iters", default=5, type=int)
parser.add_argument("--popsize", default=400, type=int)
parser.add_argument("--num_elites", default=40, type=int)
parser.add_argument("--epsilon", default=0.1, type=float)  # check default value
parser.add_argument("--alpha", default=0.1, type=float)

parser.add_argument("--n_record", default=100, type=int)
parser.add_argument("--n_eval", default=1, type=int)
# TODO: remove function references and hard code eg. obs_preproc


def get_args():
    args = parser.parse_args()

    return args
