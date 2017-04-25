import numpy as np
import scipy.stats
import os, os.path
import re
import math
import cPickle as pickle
import time
import gc

from chainer import  dataset


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
