import numpy as np
import scipy.stats
import os, os.path
import glob
import argparse
import math
from itertools import chain

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, reporter, Chain, optimizers, dataset, cuda, Variable, serializers
from chainer.training import extensions

from net import MyChain, MyChainLSTM, MyChainBiDirectionalLSTM
import stock_dataset
import stock_validation


batch_size = 128
n_epoch = 5


def save_model(model, path, device):
    if device >= 0:
        model.to_cpu()
        serializers.save_npz(path, model)
        model.to_gpu()
    else:
        serializers.save_npz(path, model)


def out_accuracy(model, data_iter, n_test_data, dst_path, device):
    @training.make_extension(trigger=(1, "epoch"))
    def make_accuracy(trainer):
        #model = trainer.updater._optimizers["main"].target
        iter_str = "%06d" % trainer.updater.iteration
        stock_validation.summarise_result(model, data_iter, n_test_data, os.path.join(dst_path, iter_str), device)
        save_model(model, os.path.join(dst_path, iter_str, "model.npz"), device)
    return make_accuracy


def get_pairs(path):
    item_list = os.path.basename(path).split("_")
    seq_size = int(item_list[1].split(".")[0])
    n_frame = (seq_size - stock_dataset.FRAME_SIZE) / stock_dataset.SHIFT_SIZE + 1
    return [(path, i) for i in range(n_frame)]


def main():

    parser = argparse.ArgumentParser(description='Stock Regression')
    parser.add_argument('data_src', default="")
    parser.add_argument('valid_idx', default="0")
    parser.add_argument('--out', default="result")
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))

    model = MyChainLSTM(in_size=9, hidden_size=5, seq_size=20, lstm_do_ratio=0.0, affine_do_ratio=0.2)
    #model = MyChainBiDirectionalLSTM(in_size=9, hidden_size=5, seq_size=20, lstm_do_ratio=0.0, affine_do_ratio=0.2)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    result_path = os.path.join(args.out, args.valid_idx)

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

    train_data = stock_dataset.MyStockDataset2(train_pairs)
    test_data = stock_dataset.MyStockDataset2(test_pairs)

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
