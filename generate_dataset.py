# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import feature as f


def check_invalid_value(data):
    x = data[:, 0]
    min_val = np.min(x)
    max_diff = np.max(np.abs(np.diff(x)))
    if max_diff > min_val * 5:
        return True
    else:
        return False


def normalize(data):
    x = data[:, 1]
    min_val = np.min(x)
    max_val = np.max(x)
    data2 = data / max_val
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

    from IPython import embed; embed()
    exit()

    return feat


def main():
    for name in os.listdir("Data/main"):
        print name
        df = pd.read_csv("Data/main/"+name, encoding="SHIFT-JIS")
        df2 = df[[u"始値", u"高値", u"安値", u"終値"]]
        data = np.array(df2)[::-1]
        if check_invalid_value(data):
            continue
        data2 = normalize(data)
        feat = calc_feature(data2)


if __name__ == "__main__":
    main()
