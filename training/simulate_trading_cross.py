# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import cPickle as pickle
import math
import argparse
import pylab as pl

from chainer import serializers, Variable

from net import MyChainLSTM

THRESH = 0.7
DAY_MAX = 3


def simulate(model, path_list):

    pred_list = []
    true_list = []

    for path in path_list:
        print path
        data = pickle.load(open(path, "rb"))
        raw_x = data[0][:, :, :9]
        raw_y = data[1][:, 0]
        x = Variable(raw_x, volatile=True)
        pred_y = model.forward(x)
        pred_y_list = pred_y.data
        pred_list.append(pred_y_list[:, 0])
        true_list.append(raw_y)

    pred_list = np.array(pred_list).T
    true_list = np.array(true_list).T

    day_n = len(true_list)

    ret_list = []
    li_list = []
    for i in range(day_n):
        pred_y = pred_list[i]
        true_y = true_list[i]
        pred_y_sorted_idx = pred_y.argsort()[::-1]
        for j in range(DAY_MAX):
            idx = pred_y_sorted_idx[j]
            if pred_y[idx] >= THRESH:
                li_list.append(pred_y[idx])
                if true_y[idx] == 1:
                    ret_list.append(1)
                else:
                    ret_list.append(0)

    ret_list = np.array(ret_list)
    li_list = np.array(li_list)

    res = [np.sum(ret_list), len(ret_list), np.sum(true_list), len(true_list.flatten())]

    return np.array(res)


def main():

    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('data_path', default="")
    parser.add_argument('result_path', default="")
    parser.add_argument('--out', '-o', default="sim_result")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    fn_list = os.listdir(args.data_path)
    fn_list.sort()
    path_list = [os.path.join(args.data_path, x) for x in fn_list[:]]

    cross_list = os.listdir(args.result_path)

    result_list = []
    is_first = True
    for cross in cross_list:
        iter_list = os.listdir(os.path.join(args.result_path, cross, "valid"))
        iter_list.sort()
        for epoch, iter_n in enumerate(iter_list):
            net_path = os.path.join(args.result_path, cross, "valid", iter_n, "model.npz")
            model = MyChainLSTM(in_size=9, hidden_size=5, seq_size=20)
            model.train = False
            serializers.load_npz(net_path, model)
            result = simulate(model, path_list)
            if is_first:
                result_list.append(result)
            else:
                result_list[epoch] += result
        is_first = False

    result_list = np.array(result_list)
    accuracy = result_list[:, 0] / result_list[:, 1]
    gt = result_list[0, 2] / result_list[0, 3]

    x = range(1, len(accuracy)+1)
    pl.figure()
    pl.bar(x, accuracy, align="center")
    pl.savefig(os.path.join(args.out, "accuracy.png"))

    accuracy = accuracy[:, np.newaxis]
    data = np.hstack([accuracy, result_list[:, :2]])
    np.savetxt(os.path.join(args.out, "result.csv"), data,
        delimiter=",", header="Ground Truth: "+str(gt), fmt="%.4f")


if __name__ == "__main__":
    main()
