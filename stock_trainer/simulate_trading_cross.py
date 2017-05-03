# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import cPickle as pickle
import math
import argparse
import pylab as pl
import glob

from chainer import serializers, Variable

from net import MyChainLSTM

THRESH = 0.85
DAY_MAX = 5


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


def run(model):

    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('data_path', default="")
    parser.add_argument('result_path', default="")
    parser.add_argument('--out', '-o', default="sim_result")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    pl.figure()

    cross_list = os.listdir(args.result_path)

    result_list = []
    is_first = True
    for cross in cross_list:

        print "cross", cross, "="*20

        iter_list = os.listdir(os.path.join(args.result_path, cross, "valid"))
        iter_list.sort()

        path_list = glob.glob(os.path.join(args.data_path, cross, "*.mat"))
        path_list.sort()

        result_epoch = []
        for epoch, iter_n in enumerate(iter_list):
            net_path = os.path.join(args.result_path, cross, "valid", iter_n, "model.npz")
            # model = MyChainLSTM(in_size=9, hidden_size=5, seq_size=20)
            serializers.load_npz(net_path, model)
            model.train = False
            result = simulate(model, path_list)
            if is_first:
                result_list.append(result)
            else:
                result_list[epoch] += result
            result_epoch.append(result)
        result_epoch = np.array(result_epoch)
        accuracy = result_epoch[:, 0] / result_epoch[:, 1]
        gt = result_epoch[0, 2] / result_epoch[0, 3]
        x = range(1, len(accuracy)+1)
        pl.bar(x, accuracy, align="center")
        pl.xlim(x[0]-1, x[-1]+1)
        pl.ylim(0, 1)
        pl.title("Ground Truth: " + "{:.2%}".format(gt))
        pl.savefig(os.path.join(args.out, "accuracy_"+cross+".png"))
        pl.clf()
        is_first = False

    result_list = np.array(result_list)
    accuracy = result_list[:, 0] / result_list[:, 1]
    gt = result_list[0, 2] / result_list[0, 3]

    x = range(1, len(accuracy)+1)
    pl.bar(x, accuracy, align="center")
    pl.title("Ground Truth: " + "{:.2%}".format(gt))
    pl.savefig(os.path.join(args.out, "accuracy_all.png"))

    accuracy = accuracy[:, np.newaxis]
    data = np.hstack([accuracy, result_list[:, :2]])
    np.savetxt(os.path.join(args.out, "result.csv"), data,
        delimiter=",", header="Ground Truth: "+str(gt), fmt="%.4f")
