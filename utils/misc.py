from random import seed as random_seed
from numpy.random import seed as np_seed
from os import environ
from tensorflow import random

def seed_everything(seed):
    random_seed(seed)
    np_seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    random.set_seed(seed + 1)