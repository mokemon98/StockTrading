import numpy as np
import scipy.stats
import os, os.path
import glob
import argparse
import math
from itertools import chain

from sklearn.metrics import classification_report

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, reporter, Chain, optimizers, dataset, cuda, Variable, serializers
from chainer.training import extensions

from net import MyChain, MyChainLSTM
import stock_dataset


batch_size = 128
n_epoch = 10
thresh = 0.7


def save_model(model, path):
    #model.to_cpu()
    serializers.save_npz(path, model)
    #model.to_gpu()


def summarise_result(model, data_iter, n_test_data, dst_path):

    data_iter.current_position = 0
    model.train = False

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    data_count = 0
    res_list = []
    while data_count + batch_size <= n_test_data:
        data = data_iter.next()
        raw_x = np.array([x[0] for x in data])
        raw_y1 = np.array([x[1] for x in data])
        #x = Variable(chainer.cuda.to_gpu(raw_x), volatile=True)
        x = Variable(raw_x, volatile=True)
        pred_y_c = model.forward(x)
        pred_y_c_list = pred_y_c.data
        for i in range(len(pred_y_c)):
            true_y_c = raw_y1[i][0]
            pred_y_c = pred_y_c_list[i][0]
            if pred_y_c > thresh:
                pred_y_c2 = 1
            else:
                pred_y_c2 = 0
            if pred_y_c > 0.5:
                pred_y_c3 = 1
            else:
                pred_y_c3 = 0
            res_list.append(np.array([true_y_c, pred_y_c2, pred_y_c, pred_y_c3], dtype=np.float16))
        data_count += batch_size

    res_list = np.array(res_list)

    report = classification_report(res_list[:, 0], res_list[:, 3], target_names=["DOWN", "UP"])
    with open(os.path.join(dst_path, "result_05.txt"), "w") as f:
        f.write("%s" % report)

    report = classification_report(res_list[:, 0], res_list[:, 1], target_names=["DOWN", "UP"])
    with open(os.path.join(dst_path, "result_07.txt"), "w") as f:
        f.write("%s" % report)

    np.savetxt(os.path.join(dst_path, "result.csv"), res_list[:, :3], fmt="%.4f",
        delimiter=",", header="true_class,pred_class,pred_li")

    model.train = True



def out_accuracy(model, data_iter, n_test_data, dst_path):
    @training.make_extension(trigger=(1, "epoch"))
    def make_accuracy(trainer):
        #model = trainer.updater._optimizers["main"].target
        iter_str = "%06d" % trainer.updater.iteration
        summarise_result(model, data_iter, n_test_data, os.path.join(dst_path, iter_str))
        save_model(model, os.path.join(dst_path, iter_str, "model.npz"))
    return make_accuracy


def get_pairs(path):
    item_list = os.path.basename(path).split("_")
    n_frame = int(item_list[1].split(".")[0])
    return [(path, i) for i in range(n_frame)]


def main():

    parser = argparse.ArgumentParser(description='Stock Regression')
    parser.add_argument('data_src', default="")
    parser.add_argument('valid_idx', default="0")
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))

    #model = MyChain()
    model = MyChainLSTM(in_size=9, hidden_size=5, seq_size=20)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    result_path = os.path.join("result", args.valid_idx)

    root = args.data_src
    fold_list = os.listdir(root)
    train_fold_list = fold_list.remove(args.valid_idx)

    train_path = []
    for fold in fold_list:
        train_dir = os.path.join(root, fold)
        train_path.extend(glob.glob(train_dir + "/*.mat"))

    test_dir = os.path.join(root, args.valid_idx)
    test_path = glob.glob(test_dir + "/*.mat")

    train_path.sort()
    test_path.sort()

    train_pairs = map(get_pairs, train_path)
    test_pairs = map(get_pairs, test_path)

    train_pairs = list(chain.from_iterable(train_pairs))
    test_pairs = list(chain.from_iterable(test_pairs))

    print "train data size : ", len(train_pairs)
    print " test data size : ", len(test_pairs)

    train_data = stock_dataset.MyStockDataset(train_pairs)
    test_data = stock_dataset.MyStockDataset(test_pairs)

    train_iter = chainer.iterators.SerialIterator(train_data, batch_size=batch_size, repeat=True, shuffle=True)
    #train_iter2 = chainer.iterators.SerialIterator(train_data, batch_size=batch_size, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test_data, batch_size=batch_size, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=result_path)

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(test_iter, eval_model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_accuracy(eval_model, test_iter, len(test_pairs), os.path.join(result_path, "valid")))
    #trainer.extend(out_accuracy(eval_model, train_iter2, len(train_pairs), "result/train"))

    trainer.run()


if __name__ == "__main__":
    main()
