from src.modeling.models.BNN import BNN
from src.modeling.trainers.BNN_trainer import BNN_trainer
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from dotmap import DotMap
from src.modeling.layers.FC_v2 import FC
from src.modeling.layers.RecalibrationLayer import RecalibrationLayer

NUM_SAMPLES = 1024
IN_DIM = 100
HIDDEN_DIM = 10
OUT_DIM = 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def stub_data():
    X = np.random.random(size=(NUM_SAMPLES, IN_DIM))
    # W_tru = np.random.random(size=(IN_DIM, OUT_DIM))
    # b_tru = 5
    # y = np.matmul(X, W_tru) + b_tru

    W_hidden = np.random.random(size=(IN_DIM, HIDDEN_DIM))
    W_last = np.random.random(size=(HIDDEN_DIM, OUT_DIM))

    y_mid = np.matmul(X, W_hidden) + 5
    y_mid[y_mid < 0] = 0
    # y_mid = sigmoid(y_mid)
    y = np.matmul(y_mid, W_last) + 2

    return (X, y)


if __name__ == "__main__":
    model_config = [
        DotMap(
            {
                "layer_name": "FC",
                "input_dim": IN_DIM,
                "output_dim": 10,
                "activation": "swish",
                "weight_decay": 0.05,
                "ensemble_size": 1,
            }
        ),
        DotMap(
            {
                "layer_name": "FC",
                "input_dim": 10,
                "output_dim": OUT_DIM,
                "activation": "swish",
                "weight_decay": 0.05,
                "ensemble_size": 1,
            }
        ),
    ]
    model = BNN(DotMap(name="test"), model_config)
    trainer_config = DotMap(
        {
            "model_dir": "random_thing",
            "epochs": 2,
            "batch_size": 2,
            "num_nets": 1,
        }
    )
    trainer = BNN_trainer(trainer_config, model)
    X, y = stub_data()
    trainer.train(X, y)
    trainer.calibrate(X, y)
