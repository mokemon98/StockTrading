# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import feature as f
import cPickle as pickle


SHIFT_SIZE = 2
PREDICT_SIZE = 5
FRAME_SIZE = 30


def check_invalid_value(data):
    if len(data) < 75 + FRAME_SIZE + PREDICT_SIZE:
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


def calc_feature(data):
    feat = data.copy()

    ma_5  = f.moving_average(data, 5)
    ma_25 = f.moving_average(data, 25)
    ma_75 = f.moving_average(data, 75)

    feat = np.hstack([feat, ma_5])
    feat = np.hstack([feat, ma_25])
    feat = np.hstack([feat, ma_75])

    bb_n, bb_p = f.bollinger_band(data, 20)

    feat = np.hstack([feat, bb_n])
    feat = np.hstack([feat, bb_p])

    hv = f.h_volatility(data, 60)

    feat = np.hstack([feat, hv])

    return feat


def save_frame(feat, out_path, stock_id):
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
        y = np.max(feat[end:end+PREDICT_SIZE, 3])
        X.append(x)
        Y.append(y)
        start += SHIFT_SIZE
    X = np.array(X)
    Y = np.array(Y)
    X = X.transpose((0, 2, 1))
    if len(X) > 50:
        train_x = X[:-50]
        train_y = Y[:-50]
        train_fn = stock_id+"_"+str(len(train_x))+".mat"
        with open(os.path.join(train_root, train_fn), "wb") as ofs:
            pickle.dump([train_x, train_y], ofs)
        test_x = X[-50:]
        test_y = Y[-50:]
        test_fn = stock_id+"_"+str(len(test_x))+".mat"
        with open(os.path.join(test_root, test_fn), "wb") as ofs:
            pickle.dump([test_x, test_y], ofs)
    else:
        train_x = X
        train_y = Y
        train_fn = stock_id+"_"+str(len(train_x))+".mat"
        with open(os.path.join(train_root, train_fn), "wb") as ofs:
            pickle.dump([train_x, train_y], ofs)


def main():
    for name in os.listdir("Data/main"):
        print name
        df = pd.read_csv("Data/main/"+name, encoding="SHIFT-JIS")
        df2 = df[[u"始値", u"高値", u"安値", u"終値", u"出来高"]]
        data = np.array(df2)[::-1]
        if check_invalid_value(data):
            continue
        data2 = normalize(data)
        feat = calc_feature(data2)
        save_frame(feat, "Data/output", name.split("-")[0])


if __name__ == "__main__":
    main()
