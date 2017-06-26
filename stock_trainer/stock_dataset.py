import numpy as np
import scipy.stats
import os, os.path
import re
import math
import cPickle as pickle
import time
import gc

from chainer import  dataset

CLOSING_AXIS = 3
FRAME_SIZE = 20
SHIFT_SIZE = 1


def normalize_zscore(data):
    data2 = data.copy()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 0.001
    data2 = (data - mean) / std
    return data2


class MyStockDataset(dataset.DatasetMixin):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        path, frame_id = self.data[i]
        with open(path, "rb") as ifs:
            stock = pickle.load(ifs)
        x = stock[0][frame_id][:, :9]
        x = x.reshape((x.shape[0], 9))
        #x = x.transpose((1, 0))
        y = stock[1][frame_id]
        y1 = y[0].reshape(1)
        y1 = y1.astype(np.int32)
        #y2 = y[1].reshape(1)
        return x, y1


class MyStockDataset2(dataset.DatasetMixin):

    def __init__(self, data, using_y_axis):
        self.data = data
        self.y_axis = using_y_axis

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        path, frame_id = self.data[i]
        with open(path, "rb") as ifs:
            stock = pickle.load(ifs)
        start = frame_id * SHIFT_SIZE
        end = start + FRAME_SIZE
        x = stock[0][start:end, :9]
        base_x = x[0, CLOSING_AXIS]
        x = x / base_x
        mean = np.mean(x[:, CLOSING_AXIS])
        std = np.std(x[:, CLOSING_AXIS]) + 0.001
        x2 = (x - mean) / std
        #x2 = normalize_zscore(x)
        y = stock[1][end-1][self.y_axis]  # 0: 0.0%, 1: 0.1%, 2: 0.2%, 3: rate
        y2 = y.reshape(len(y))
        y2 = y2.astype(np.int32)
        rate = stock[1][end-1][-1]
        rate = rate.reshape(1)
        return x2, y2, rate


class MyStockDataset3(dataset.DatasetMixin):

    def __init__(self, data, using_y_axis):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        path, frame_id = self.data[i]
        with open(path, "rb") as ifs:
            stock = pickle.load(ifs)
        start = frame_id * SHIFT_SIZE
        end = start + FRAME_SIZE
        x = stock[0][start:end, :9]
        base_x = x[0, CLOSING_AXIS]
        x = x / base_x
        mean = np.mean(x[:, CLOSING_AXIS])
        std = np.std(x[:, CLOSING_AXIS]) + 0.001
        x2 = (x - mean) / std
        #x2 = normalize_zscore(x)
        #y = stock[1][end-1][self.y_axis]  # 0: 0.0%, 1: 0.1%, 2: 0.2%, 3: rate
        #y2 = y.reshape(len(y))
        #y2 = y2.astype(np.int32)
        rate = stock[1][end-1][-1]
        rate = rate.reshape(1)
        if rate < -0.005:
            y = 0
        elif rate < 0.005:
            y = 1
        else:
            y = 2
        y2 = np.array([y], dtype=np.int32)
        y2 = y2.reshape(1)
        return x2, y2, rate