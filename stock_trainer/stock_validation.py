import numpy as np
import scipy.stats
import os, os.path
import math
import pylab as plt
from itertools import chain, product

from sklearn.metrics import classification_report, confusion_matrix

import chainer
from chainer import Variable

from net import MyChain, MyChainLSTM, MyChainBiDirectionalLSTM
import stock_dataset


thresh = 0.001


def plot_confusion_matrix(cm, classes, path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)

    plt.close()


def summarise_result(model, data_iter, n_test_data, dst_path, device):

    data_iter.current_position = 0

    batch_size = data_iter.batch_size

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    data_count = 0
    true_list = []
    li_list = []
    while data_count + batch_size <= n_test_data:
        data = data_iter.next()
        raw_x = np.array([x[0] for x in data])
        raw_y = np.array([x[1] for x in data])
        if device >= 0:
            x = Variable(chainer.cuda.to_gpu(raw_x), volatile=True)
            pred_y_c = model.forward(x)
            pred_y_c = chainer.duda.to_cpu(pred_y_c.data)
        else:
            x = Variable(raw_x, volatile=True)
            pred_y_c = model.forward(x)
            pred_y_c = pred_y_c.data
        true_list.extend(raw_y.flatten().tolist())
        li_list.extend(pred_y_c.flatten().tolist())
        data_count += batch_size

    true_list = np.array(true_list)
    li_list = np.array(li_list)

    pred_list = [1 if x >= 0.5 else 0 for x in li_list]
    li_sorted_idx = li_list.argsort()[::-1]

    true_list = true_list[li_sorted_idx]
    li_list = li_list[li_sorted_idx]
    pred_list = np.array(pred_list)[li_sorted_idx]

    # output result of thresh 0.5
    report = classification_report(true_list, pred_list, target_names=["DOWN", "UP"], labels=[0, 1])
    with open(os.path.join(dst_path, "confusion_all.txt"), "w") as f:
        f.write("%s" % report)

    cm = confusion_matrix(true_list, pred_list, labels=[0, 1])
    plot_confusion_matrix(cm , ["0", "1"], os.path.join(dst_path, "confusion_all.png"))

    # output result of higher likelihood
    n = int(len(true_list) * thresh)
    report = classification_report(true_list[:n], pred_list[:n], target_names=["DOWN", "UP"], labels=[0, 1])
    with open(os.path.join(dst_path, "confusion_top.txt"), "w") as f:
        f.write("%s" % report)

    cm = confusion_matrix(true_list[:n], pred_list[:n], labels=[0, 1])
    plot_confusion_matrix(cm , ["0", "1"], os.path.join(dst_path, "confusion_top.png"))

    # output all result
    true_list = np.expand_dims(true_list, 1)
    pred_list = np.expand_dims(pred_list, 1)
    li_list = np.expand_dims(li_list, 1)
    np.savetxt(os.path.join(dst_path, "result.csv"), np.hstack([true_list, pred_list, li_list]),
        fmt="%.4f", delimiter=",", header="true_class,pred_class,pred_li")
