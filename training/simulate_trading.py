# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import cPickle as pickle
import math
import argparse
import time

from chainer import serializers, Variable

from net import MyChainLSTM

THRESH = 0.7
DAY_MAX = 3


def simulate(model, path_list, out_path):

    pred_list = []
    true_list = []

    t2 = time.time()

    for path in path_list:
        t1 = time.time()
        print path, t1 - t2
        data = pickle.load(open(path, "rb"))
        raw_x = data[0][:, :, :9]
        raw_y = data[1][:, 0]
        x = Variable(raw_x, volatile=True)
        pred_y = model.forward(x)
        pred_y_list = pred_y.data
        pred_list.append(pred_y_list[:, 0])
        true_list.append(raw_y)
        t2 = t1

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

    print np.sum(ret_list), "/", len(ret_list), "(", np.mean(ret_list), ")"
    print np.sum(true_list), "/", len(true_list.flatten()), "(", np.mean(true_list), ")"

    ret_data = np.hstack([ret_list.reshape(len(ret_list), 1), li_list.reshape(len(li_list), 1)])
    np.savetxt(os.path.join(out_path, "result.csv"), ret_data, delimiter=",")


def main():

    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('valid_data', default="")
    parser.add_argument('net', default="")
    parser.add_argument('--out', '-o', default="sim_result")
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    fn_list = os.listdir(args.valid_data)
    fn_list.sort()
    path_list = [os.path.join(args.valid_data, x) for x in fn_list[:]]

    model = MyChainLSTM(in_size=9, hidden_size=5, seq_size=20)
    model.train = False
    serializers.load_npz(args.net, model)

    simulate(model, path_list, args.o)



if __name__ == "__main__":
    main()
