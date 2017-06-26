# -*- coding: utf-8 -*-

import os, os.path
import numpy as np
import cPickle as pickle
import argparse
import glob

from chainer import serializers, Variable


CLOSING_AXIS = 3
FRAME_SIZE = 20
SHIFT_SIZE = 1


def get_frames(sequence):
    N = (len(sequence) - FRAME_SIZE) / SHIFT_SIZE + 1
    X = []
    for j in range(N):
        start = j * SHIFT_SIZE
        end = j + FRAME_SIZE
        x = sequence[start:end]
        base_x = x[0, CLOSING_AXIS]
        x = x / base_x
        mean = np.mean(x[:, CLOSING_AXIS])
        std = np.std(x[:, CLOSING_AXIS]) + 0.001
        x2 = (x - mean) / std
        X.append(x2)
    return np.array(X)


def predict_sequences(model, path_list, dst_path):

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for path in path_list:

        print path

        data = pickle.load(open(path, "rb"))
        raw_x = data[0][:, :9]
        raw_y = data[1]

        x = get_frames(raw_x)
        y = raw_y[FRAME_SIZE-1:]

        x2 = Variable(x, volatile=True)
        pred_y = model.forward(x2)
        pred_y = pred_y.data

        core_name = os.path.basename(path).split(".")[0]
        np.savetxt(os.path.join(dst_path, core_name+".csv"),
            np.hstack([pred_y, y]),
            fmt="%.4f", delimiter=",", header="pred0,pred1,pred2,true0,true1,true2,rate")


def run(model):

    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('data_path', default="")
    parser.add_argument('result_path', default="")
    parser.add_argument('--out', '-o', default="predict")
    args = parser.parse_args()

    cross_list = os.listdir(args.result_path)

    result_list = []
    is_first = True
    for cross in cross_list:

        print "cross", cross, "="*20

        iter_list = os.listdir(os.path.join(args.result_path, cross, "valid"))
        iter_list.sort()

        path_list = glob.glob(os.path.join(args.data_path, cross, "*.mat"))
        path_list.sort()

        for epoch, iter_n in enumerate(iter_list):
            net_path = os.path.join(args.result_path, cross, "valid", iter_n, "model.npz")
            serializers.load_npz(net_path, model)
            model.train = False
            dst_path = os.path.join(args.out, cross, "epoch_"+str(epoch+1))
            predict_sequences(model, path_list, dst_path)
