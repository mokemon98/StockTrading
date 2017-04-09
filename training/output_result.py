import numpy as np
import cv2
import os, os.path
import glob
import argparse
import math

import pylab as plt

import chainer
from chainer import cuda, Variable, serializers

from net import MyChain
import joint_info as ji
import image_dataset


batch_size = 32


def update_accuracy_count(accuracy_count, pos_list):
    for i in range(len(pos_list)):
        pos = pos_list[i]
        l = math.sqrt( (pos[0] - pos[2])**2 + (pos[1] - pos[3])**2 )
        for j in range(len(accuracy_count)):
            if l <= j:
                accuracy_count[i, j] += 1


def calc_accuracy(accuracy_count, n_data, dst_path):
    acc_path = os.path.join(dst_path, "accuracy")
    if not os.apth.exists(acc_path):
        os.makedirs(acc_path)
    accuracy = accuracy_count / n_data
    np.savetxt(os.path.join(acc_path, "accuracy.csv"))
    for i in range(len(accuracy_count)):
        joint_name = ji.joint_name_list[i+1]
        plt.plot(accuracy[i], "-")
        plt.title(joint_name)
        plt.xlabel("distance [pixel]")
        plt.ylabel("accuracy")
        plt.savefig(os.path.join(acc_path, joint_name+".png"))
        plt.clf()
    accuracy_mean = np.mean(accuracy, axis=0)
    plt.plot(accuracy_mean, "-")
    plt.title("average")
    plt.xlabel("distance [pixel]")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(acc_path, "average.png"))
    plt.clf()


def output_result(model, data_iter, n_test_data, dst_path):
    data_count = 0
    accuracy_count = np.zeros((ji.n_joint-1, 20))
    while data_count < n_test_data:
        data = data_iter.next()
        raw_x = np.array([x[0] for x in data])
        raw_y = np.array([x[3] for x in data])
        info = [x[2] for x in data]
        x = Variable(chainer.cuda.to_gpu(raw_x), volatile=True)
        y = model.forward(x)
        pred_y = y.data
        for i in range(len(pred_y)):
            pred_joint_list = []
            for j, joint_map in enumerate(pred_y[i][1:]):
                use_joint_id = ji.use_joint[j]
                joint_map = chainer.cuda.to_cpu(joint_map)
                joint_map2 = cv2.resize(joint_map, (image_dataset.in_size, image_dataset.in_size))
                pos_y, pos_x = np.unravel_index(joint_map2.argmax(), joint_map2.shape)
                true_y = raw_y[i][use_joint_id] * image_dataset.in_size
                pred_joint_list.append([true_y[0], true_y[1], pos_x, pos_y])
            update_accuracy_count(accuracy_count, pred_joint_list)
            path = os.path.join(dst_path, info[i][0], info[i][1], info[i][2])
            if not os.path.exists(path):
                os.makedirs(path)
            pred_joint_list = np.array(pred_joint_list)
            np.savetxt(os.path.join(path, info[i][3]+".csv"), pred_joint_list, delimiter=",", fmt="%d")
            print os.path.join(path, info[i][3]+".csv")
        calc_accuracy(accuracy_count, n_test_data, dst_path)
        data_count += len(data)

def main():

    parser = argparse.ArgumentParser(description='Chainer example: MNIST')

    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-o', default='result/model.npz',
                        help='Directory to output the result')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))

    model = MyChain(ji.n_joint)
    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    root = "/home/Keita.Mochizuki/LocalWork/MotionCapture/Data"

    test_x_dir = os.path.join(root, "test_x")
    test_y_dir = os.path.join(root, "test_y")
    test_x = glob.glob(test_x_dir + "/*/*/*/*.pgm")
    test_y = glob.glob(test_y_dir + "/*/*/*/*.csv")
    n_test_data = len(test_y)

    test_x.sort()
    test_y.sort()

    print " test data size : ", len(test_x), len(test_y)

    test_data = image_dataset.MyImageDatasetForTest(test_x, test_y)
    test_iter = chainer.iterators.SerialIterator(test_data, batch_size=batch_size, repeat=False, shuffle=False)

    #output_result(model, test_iter, n_test_data, "result/output")
    output_result(model, test_iter, 500, "result/output")


if __name__ == "__main__":
    main()
