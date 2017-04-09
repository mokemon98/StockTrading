import os
import argparse
import pylab as pl
import json


def get_error(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    error_list = []
    for d in dir_list:
        with open(os.path.join(path, d, "result.txt"), "r") as ifs:
            line = ifs.readline()[:-1]
            v = float(line)
            error_list.append(v)
    return error_list


def get_loss(path):
    train_loss = []
    test_loss = []
    with open(os.path.join(path, "log"), "r") as ifs:
        data = json.load(ifs)
        for res in data:
            train_loss.append(res["main/loss"])
            test_loss.append(res["validation/main/loss"])
    return train_loss, test_loss


def plot_result(train_error, test_error, train_loss, test_loss):
    fig, host = pl.subplots()
    par1 = host.twinx()
    p1, = host.plot(train_loss, "b-", label="train loss")
    p2, = host.plot(test_loss, "r-", label="test loss")
    p3, = par1.plot(train_error, "b--", label="train_error")
    p4, = par1.plot(test_error, "r--", label="test_error")
    lines = [p1, p2, p3, p4]
    host.legend(lines, [l.get_label() for l in lines])
    pl.show()

def main():
    parser = argparse.ArgumentParser(description='Stock Regression')
    parser.add_argument('src', default="result", help="result path")
    args = parser.parse_args()

    train_path = os.path.join(args.src, "train")
    test_path = os.path.join(args.src, "valid")

    train_error = get_error(train_path)
    test_error = get_error(test_path)

    train_loss, test_loss = get_loss(args.src)

    plot_result(train_error, test_error, train_loss, test_loss)


if __name__ == "__main__":
    main()
