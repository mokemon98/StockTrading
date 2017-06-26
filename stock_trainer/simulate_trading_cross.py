# -*- coding: utf-8 -*-

import pandas as pd
import os, os.path
import numpy as np
import cPickle as pickle
import math
import argparse
import pylab as pl
import glob
import seaborn as sns

from chainer import serializers, Variable

from net import MyChainLSTM

TOP_RATIO = 0.001
MAX_STOCK = 5


def read_data(path_list):
    pred0 = []
    pred1 = []
    pred2 = []
    rate = []
    for data_path in path_list:
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        pred0.append(np.expand_dims(data[:, 0], 1))
        pred1.append(np.expand_dims(data[:, 1], 1))
        pred2.append(np.expand_dims(data[:, 2], 1))
        rate.append(np.expand_dims(data[:, 6], 1))
    pred0 = np.hstack(pred0)
    pred1 = np.hstack(pred1)
    pred2 = np.hstack(pred2)
    rate = np.hstack(rate)
    return pred0, pred1, pred2, rate


def run():

    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('predict_path', default="")
    parser.add_argument('--out', '-o', default="sim_result")
    args = parser.parse_args()

    cross_list = os.listdir(args.predict_path)

    summary_df = pd.DataFrame()

    all_rate = []

    for cross in cross_list:

        print "cross", cross, "="*20

        out_root = os.path.join(args.out, cross)

        if not os.path.exists(out_root):
            os.makedirs(out_root)

        epoch_list = os.listdir(os.path.join(args.predict_path, cross))
        epoch_list.sort()

        for epoch in epoch_list:
            path_list = glob.glob(os.path.join(args.predict_path, cross, epoch, "*.csv"))
            path_list.sort()

            pred0, pred1, pred2, rate = read_data(path_list)

            all_rate.append(rate)

            def calc_result(pred, rate):

                def get_thresh(pred):
                    pred2 = pred.flatten()
                    pred2.sort()
                    pred2 = pred2[::-1]
                    n = int(len(pred2) * TOP_RATIO)
                    th = pred2[n]
                    return th

                th = get_thresh(pred)

                def get_stock(daily_pred, th):
                    daily_pred2 = np.sort(daily_pred)[::-1]
                    li = [x for x in daily_pred2 if x >= th]
                    li = li[:min(len(li), MAX_STOCK)]
                    idx = [daily_pred2.tolist().index(x) for x in li]
                    return idx

                stock = map(lambda x: get_stock(x, th), pred)
                res = [x[y] for x, y in zip(rate, stock)]

                count_tmp = [len(x) for x in res]
                count = reduce(lambda x, y: x+y, count_tmp)
                total_tmp = [np.sum(x) for x in res]
                total = reduce(lambda x, y: x+y, total_tmp)
                avg = total / count

                return res, avg, count

            tmp = map(lambda x: calc_result(x, rate), [pred0, pred1, pred2])

            res_list = [x[0] for x in tmp]
            avg_list = [x[1] for x in tmp]
            count_list = [x[2] for x in tmp]

            epoch_str = epoch.split("_")[1]
            fn0 = "cross"+cross+"_epoch"+epoch_str+"_res0.csv"
            fn1 = "cross"+cross+"_epoch"+epoch_str+"_res1.csv"
            fn2 = "cross"+cross+"_epoch"+epoch_str+"_res2.csv"
            out_path_list = [os.path.join(out_root, fn) for fn in [fn0, fn1, fn2]]

            def out_result(res, avg, out_path):
                with open(out_path, "w") as ofs:
                    ofs.write("average,%f\n" % avg)
                    [ofs.write("%s\n" % str(x)) for x in res]

            map(out_result, res_list, avg_list, out_path_list)

            df = pd.DataFrame([
                            [cross, epoch, 0, avg_list[0], count_list[0]],
                            [cross, epoch, 1, avg_list[1], count_list[1]],
                            [cross, epoch, 2, avg_list[2], count_list[2]]],
                            columns=["cross", "epoch", "method", "result", "count"])

            summary_df = summary_df.append(df)

    avg_rate = np.mean(np.array(all_rate))

    def out_figure(df, method, out_path):
        df2 = df[df["method"]==method]
        count = np.mean(np.array(df2["count"]))
        pl.figure()
        sns.barplot(x="epoch", y="result", data=df)
        title = "count:  %.1f\navg:  %.4f" % (count, avg_rate)
        pl.title(title)
        pl.savefig(out_path)

    fig0 = os.path.join(args.out, "result0.png")
    fig1 = os.path.join(args.out, "result1.png")
    fig2 = os.path.join(args.out, "result2.png")
    map(lambda x, y: out_figure(summary_df, x, y), [0, 1, 2], [fig0, fig1, fig2])
