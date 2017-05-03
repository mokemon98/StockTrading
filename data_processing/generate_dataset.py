# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import feature as f
import cPickle as pickle
import math
import argparse

import config as c


SHIFT_SIZE = 1
PREDICT_SIZE = 1
FRAME_SIZE = 20
TEST_SIZE = 180
CLOSING_AXIS = 3
YIELD_AXIS = 4
CROSS_N = 5
MIN_SIZE = 1052
#MIN_SIZE = TEST_SIZE * CROSS_N + 75 + FRAME_SIZE + PREDICT_SIZE


def check_invalid_value(data):
    if len(data) < MIN_SIZE:
        return True  # invalid
    x = data[:, CLOSING_AXIS]
    diff = np.abs(np.diff(x))
    rate = diff / x[:-1]
    max_rate = np.max(rate)
    if max_rate > 0.2:
        return True  # invalid
    else:
        return False  # valid


def normalize_zscore(data):
    data2 = data.copy()
    mean = np.mean(data[75:, CLOSING_AXIS])
    std = np.std(data[75:, CLOSING_AXIS])
    for i in range(data.shape[1]):
        data2[:, i] = (data[:, i] - mean) / std
    return data2


def normalize_zscore2(data):
    data2 = data.copy()
    for i in range(data.shape[1]):
        mean2 = np.mean(data[75:, i])
        std2 = np.std(data[75:, i])
        data2[:, i] = (data[:, i] - mean2) / std2
    return data2


def calc_feature(data):
    feat = data.copy()

    ma_5  = f.moving_average(data, 5)
    ma_25 = f.moving_average(data, 25)
    ma_75 = f.moving_average(data, 75)
    feat = np.hstack([feat, ma_5, ma_25, ma_75])

    bb_n, bb_p = f.bollinger_band(data, 20)
    feat = np.hstack([feat, bb_n, bb_p])

    #hv = f.h_volatility(data, 60)
    #feat = np.hstack([feat, hv])

    return feat


def get_label_and_value_on_next_day(feat, closing, end):
    c_1 = closing[end-1]
    c_2 = closing[end]
    d = (c_2 - c_1) / c_1
    #if abs(d) > 0.05:
    #    y = None
    if d > 0:
        y = [1, d]
    else:
        y = [0, d]
    return y


def save_frame(feat, closing, out_root, stock_id):

    start = 75
    L = len(feat)
    X = []
    Y = []
    while start + FRAME_SIZE + PREDICT_SIZE <= L:
        end = start+FRAME_SIZE
        x = feat[start:end]
        y = get_label_and_value_on_next_day(feat, closing, end)
        if y is not None:
            X.append(x)
            Y.append(y)
        start += SHIFT_SIZE

    if len(X) == 0: return

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    for i in range(CROSS_N):

        out_path = os.path.join(out_root, str(i+1))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        start = i * TEST_SIZE
        end = (i + 1) * TEST_SIZE

        data_x = X[start:end]
        data_y = Y[start:end]
        data_fn = stock_id+"_"+str(len(data_x))+".mat"

        with open(os.path.join(out_path, data_fn), "wb") as ofs:
            pickle.dump([data_x, data_y], ofs, protocol=2)


def save_sequence(feat, closing, out_root, stock_id):

    start = 75
    L = len(feat)

    diff = np.diff(closing)
    rate = diff / closing[:-1]

    Y1 = [1 if x > 0 else 0 for x in diff.tolist()]
    Y2 = [1 if x >= 0.01 else 0 for x in rate.tolist()]
    Y3 = [1 if x >= 0.02 else 0 for x in rate.tolist()]

    Y = np.array([Y1, Y2, Y3, rate], dtype=np.float32).T

    X = np.array(feat[start:-1], dtype=np.float32)
    Y = np.array(Y[start:], dtype=np.float32)

    for i in range(CROSS_N):

        out_path = os.path.join(out_root, str(i+1))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        start = i * TEST_SIZE
        end = (i + 1) * TEST_SIZE + FRAME_SIZE - 1

        if end < len(X):
            data_x = X[start:end]
            data_y = Y[start:end]
            data_fn = stock_id+"_"+str(len(data_x))+".mat"

            with open(os.path.join(out_path, data_fn), "wb") as ofs:
                pickle.dump([data_x, data_y], ofs, protocol=2)


def get_indices():
    indices = None
    root = os.path.join(c.data_root, "indices")
    path_list = os.listdir(root)
    path_list.sort()
    for name in path_list:
        df = pd.read_csv(os.path.join(root, name), encoding="SHIFT-JIS")
        df2 = df[[u"終値"]]
        if indices is None:
            indices = np.array(df2)
        else:
            indices = np.hstack([indices, np.array(df2)])
    return indices


def main():

    parser = argparse.ArgumentParser(description='Stock Regression')
    parser.add_argument('dst', default="")
    args = parser.parse_args()

    indices = get_indices()
    indices = indices[::-1, :2]

    df_ind = pd.read_csv(os.path.join(c.data_root, "indices", u"I101_日経平均株価.csv"), encoding="SHIFT-JIS")
    df_ind2 = df_ind[[u"日付"]]

    root = os.path.join(c.data_root, "main")
    for name in os.listdir(root):
        df = pd.read_csv(os.path.join(root, name), encoding="SHIFT-JIS")
        df2 = df[[u"始値", u"高値", u"安値", u"終値", u"出来高"]]
        df3 = df[[u"日付"]]
        data = np.array(df2)
        flag = False
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if math.isnan(data[i, j]):
                    flag = True
                    data = data[:i]
                    break
            if flag:
                break
        data = data[::-1]
        if check_invalid_value(data):
            continue
        print name, len(data)
        if df_ind2.ix[len(data)-1].values[0] != df3.ix[len(data)-1].values[0]:
            print "Day Matching Error !!!"
            print df_ind2.ix[len(data)-1].values[0], df3.ix[len(data)-1].values[0]
            exit()
        feat1 = calc_feature(data[:, :YIELD_AXIS])
        #feat1_2 = normalize_zscore(feat1)
        feat2 = np.hstack([indices[-len(data):], data[:, YIELD_AXIS].reshape((len(data), 1))])
        #feat2_2 = normalize_zscore2(feat2)
        feat3 = np.hstack([feat1, feat2])
        #save_frame(feat3, data[:, CLOSING_AXIS], args.dst, name.split("-")[0])
        save_sequence(feat3, data[:, CLOSING_AXIS], args.dst, name.split("-")[0])


if __name__ == "__main__":
    main()
