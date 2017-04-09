# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import feature as f
import cPickle as pickle
import math


SHIFT_SIZE = 2
PREDICT_SIZE = 1
FRAME_SIZE = 30
TEST_SIZE = 30
MIN_SIZE = 400


def check_invalid_value(data):
    #if len(data) < 75 + FRAME_SIZE + PREDICT_SIZE:
    if len(data) < MIN_SIZE:
        return True
    x = data[:, 0]
    min_val = np.min(x)
    max_diff = np.max(np.abs(np.diff(x)))
    if max_diff > min_val * 5:
        return True
    else:
        return False


def normalize(data):
    data2 = data.copy()
    x = data[:, 1]
    max_val = np.max(x)
    data2[:, :4] = data[:, :4] / max_val
    x = data[:, 4]
    max_val = np.max(x)
    data2[:, 4] = data[:, 4] / max_val
    return data2


def normalize_zscore(data):
    data2 = data.copy()
    for i in range(data.shape[1]):
        mean = np.mean(data[75:, i])
        std = np.std(data[75:, i])
        data2[:, i] = (data[:, i] - mean) / std
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
    if abs(d) > 0.05:
        y = None
    elif d > 0:
        y = [1, d]
    else:
        y = [0, d]
    return y


def save_frame(feat, closing, out_path, stock_id):

    train_root = os.path.join(out_path, "train")
    test_root = os.path.join(out_path, "test")
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    if not os.path.exists(test_root):
        os.makedirs(test_root)

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
    X = X.transpose((0, 2, 1))
    if len(X) > TEST_SIZE:
        train_x = X[:-TEST_SIZE]
        train_y = Y[:-TEST_SIZE]
        train_fn = stock_id+"_"+str(len(train_x))+".mat"
        with open(os.path.join(train_root, train_fn), "wb") as ofs:
            pickle.dump([train_x, train_y], ofs, protocol=2)
        test_x = X[-TEST_SIZE:]
        test_y = Y[-TEST_SIZE:]
        test_fn = stock_id+"_"+str(len(test_x))+".mat"
        with open(os.path.join(test_root, test_fn), "wb") as ofs:
            pickle.dump([test_x, test_y], ofs, protocol=2)
    else:
        train_x = X
        train_y = Y
        train_fn = stock_id+"_"+str(len(train_x))+".mat"
        with open(os.path.join(train_root, train_fn), "wb") as ofs:
            pickle.dump([train_x, train_y], ofs, protocol=2)


def main():
    for name in os.listdir("Data/main"):
        #if not name.startswith("2924"):
        #    continue
        print name
        df = pd.read_csv("Data/main/"+name, encoding="SHIFT-JIS")
        df2 = df[[u"始値", u"高値", u"安値", u"終値", u"出来高"]]
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
        feat = calc_feature(data)
        feat2 = normalize_zscore(feat)
        save_frame(feat2, data[:, 3], "Data/output_class_and_rate_3", name.split("-")[0])


if __name__ == "__main__":
    main()
