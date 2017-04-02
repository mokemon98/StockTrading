# -*- coding: utf-8 -*-

import numpy as np


def moving_average(data, win_size):
    weights = np.repeat(1.0, win_size) / win_size
    sma = np.convolve(data[:, 3], weights, "valid")
    sma = sma.reshape((len(sma), 1))
    pre = data[:win_size-1, 3].reshape((win_size-1, 1))
    sma2 = np.vstack([pre, sma])
    return sma2


def bollinger_band(data, win_size):
    x = data[:, 3]
    l = len(x)
    bb_std = map(lambda i: np.std(x[i:i+win_size]), range(l-win_size+1))
    bb_std = np.array(bb_std)
    bb_std = bb_std.reshape((len(bb_std), 1))
    bb_std2 = np.vstack([np.zeros((win_size-1, 1)), bb_std])
    bb_ma = moving_average(data, win_size)
    return bb_ma+bb_std2*2, bb_ma-bb_std2*2


def h_volatility(data, win_size):
    x = data[:, 3]
    x2 = np.log(x[1:] / x[:-1])
    l = len(x2)
    std = map(lambda i: np.std(x2[i:i+win_size]), range(l-win_size+1))
    std = np.array(std)
    std = std.reshape((len(std), 1))
    std2 = np.vstack([np.zeros((win_size, 1)), std])
    return std2
