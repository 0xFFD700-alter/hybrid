import os
import torch
import random
import numpy as np
import torch.multiprocessing
from fedlab.utils.functional import setup_seed

class Recoder(object):
    def __init__(self):
        self.last = 0
        self.values = []
        self.nums = []

    def update(self, val, n=1):
        self.last = val
        self.values.append(val)
        self.nums.append(n)

    def avg(self):
        sum = np.sum(np.asarray(self.values) * np.asarray(self.nums))
        count = np.sum(np.asarray(self.nums))
        return sum / count


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    setup_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_point(epoch, epochs, model, accuracy_list, save_model_interval, save_accuracy_interval):
    if epoch % save_model_interval == 0 or epoch == epochs:
        torch.save(model.state_dict(), f'models/model' + str(epoch) + '.pth',_use_new_zipfile_serialization=False)
    if epoch % save_accuracy_interval == 0 or epoch == epochs:
        np.save(f'results/accuracy_list.npy', np.asarray(accuracy_list))
