# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import cPickle as pickle
import math
import argparse

from chainer import serializers, Variable

from net import MyChainLSTM2


def simulate(model, path_list, out_path):

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

    max_li_idx_list = np.argmax(pred_list, axis=1)
    max_li_list = np.max(pred_list, axis=1)

    day_n = len(true_list)

    ret_list = []
    for i in range(day_n):
        max_li_idx = max_li_idx_list[i]
        t = true_list[i, max_li_idx]
        if t == 1:
            ret_list.append(1)
        else:
            ret_list.append(0)

    ret_list = np.array(ret_list)

    print np.sum(ret_list), "/", len(ret_list), "(", np.mean(ret_list), ")"
    print np.sum(true_list), "/", len(true_list.flatten()), "(", np.mean(true_list), ")"

    ret_data = np.hstack([ret_list.reshape(day_n, 1), max_li_list.reshape(day_n, 1)])
    np.savetxt(os.path.join(out_path, "result.csv"), ret_data, delimiter=",")


def main():

    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('valid_data', default="")
    parser.add_argument('net', default="")
    parser.add_argument('-o', default="sim_result")
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    fn_list = os.listdir(args.valid_data)
    fn_list.sort()
    path_list = [os.path.join(args.valid_data, x) for x in fn_list[:]]

    model = MyChainLSTM2(in_size=9, hidden_size=5, seq_size=20)
    model.train = False
    serializers.load_npz(args.net, model)

    simulate(model, path_list, args.o)



if __name__ == "__main__":
    main()
