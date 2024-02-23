import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import mean


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mean_absolute_percentage_error(y_true, y_pred):
    return mean(abs((y_true - y_pred) / (y_true + 1e-8))) * 100


# print(mean_absolute_percentage_error(np.array([100, 200, 300, 400]), np.array([110, 190, 305, 390])))
