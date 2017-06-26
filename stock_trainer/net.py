import math
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter, Chain, Variable


class MyChainLSTM(Chain):

    def __init__(self, in_size, hidden_size, seq_size, out_size,
                    device, lstm_do_ratio=0.0, affine_do_ratio=0.5):
        super(MyChainLSTM, self).__init__(
            lstm1 = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            l1 = L.Linear(hidden_size*seq_size, 50),
            l2 = L.Linear(50, out_size)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.affine_do_ratio = affine_do_ratio
        self.device = device
        self.layer_n = 1

    def __call__(self, xs, ts, rate):
        os = self.predict(xs)
        loss = F.sigmoid_cross_entropy(os, ts)
        self.loss = loss
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, xs):
        os = self.predict(xs)
        os2 = F.sigmoid(os)
        return os2

    def predict(self, x):
        x2 = [item for item in x]
        if self.device >= 0:
            init = np.zeros((self.layer_n, len(x2), self.hidden_size), dtype=np.float32)
            hx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
            cx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
        else:
            hx = None
            cx = None
        h = self.lstm1(hx, cx, x2, train=self.train)[2]
        h = [F.expand_dims(item, 0) for item in h]
        h = F.concat(h, 0)
        h = F.dropout(F.relu(self.l1(h)), train=self.train, ratio=self.affine_do_ratio)
        o = self.l2(h)
        return o


class MyChainBiDirectionalLSTM(Chain):

    def __init__(self, in_size, hidden_size, seq_size, out_size,
                    device, lstm_do_ratio=0.0, affine_do_ratio=0.5):
        super(MyChainBiDirectionalLSTM, self).__init__(
            lstm_f = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            lstm_b = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            l1 = L.Linear(hidden_size*seq_size*2, 50),
            l2 = L.Linear(50, out_size)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.affine_do_ratio = affine_do_ratio
        self.device = device
        self.layer_n = 1

    def __call__(self, xs, ts, ratio):
        os = self.predict(xs)
        loss = F.sigmoid_cross_entropy(os, ts)
        self.loss = loss
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, xs):
        os = self.predict(xs)
        os2 = F.sigmoid(os)
        return os2

    def predict(self, xs):
        xs_f = [x for x in xs]
        xs_b = [x[::-1] for x in xs]

        if self.device >= 0:
            init = np.zeros((self.layer_n, len(xs_f), self.hidden_size), dtype=np.float32)
            hx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
            cx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
        else:
            hx = None
            cx = None

        hs_f = self.lstm_f(hx, cx, xs_f, train=self.train)[2]
        hs_f = [F.expand_dims(h, 0) for h in hs_f]
        hs_f = F.concat(hs_f, 0)

        hs_b = self.lstm_b(hx, cx, xs_b, train=self.train)[2]
        hs_b = [F.expand_dims(h, 0) for h in hs_b]
        hs_b = F.concat(hs_b, 0)

        hs = F.concat((hs_f, hs_b[::-1]), 2)
        hs = F.dropout(F.relu(self.l1(hs)), train=self.train, ratio=self.affine_do_ratio)
        os = self.l2(hs)

        return os


class MyChainLSTM2(Chain):

    def __init__(self, in_size, hidden_size, seq_size, out_size,
                    device, lstm_do_ratio=0.0, affine_do_ratio=0.5):
        super(MyChainLSTM2, self).__init__(
            lstm1 = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            l1 = L.Linear(hidden_size*seq_size, 10),
            l2 = L.Linear(10, out_size)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.affine_do_ratio = affine_do_ratio
        self.device = device
        self.layer_n = 1

    def __call__(self, xs, ts, rate):
        os = self.predict(xs)
        loss = F.sigmoid_cross_entropy(os, ts)
        self.loss = loss
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, xs):
        os = self.predict(xs)
        os2 = F.sigmoid(os)
        return os2

    def predict(self, x):
        x2 = [item for item in x]
        if self.device >= 0:
            init = np.zeros((self.layer_n, len(x2), self.hidden_size), dtype=np.float32)
            hx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
            cx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
        else:
            hx = None
            cx = None
        h = self.lstm1(hx, cx, x2, train=self.train)[2]
        h = [F.expand_dims(item, 0) for item in h]
        h = F.concat(h, 0)
        h = F.dropout(F.relu(self.l1(h)), train=self.train, ratio=self.affine_do_ratio)
        o = self.l2(h)
        return o


class MyChainLSTM3(Chain):

    def __init__(self, in_size, hidden_size, seq_size, out_size,
                    device, lstm_do_ratio=0.0, affine_do_ratio=0.5):
        super(MyChainLSTM3, self).__init__(
            lstm1 = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            l1 = L.Linear(hidden_size*seq_size, 50),
            l2 = L.Linear(50, out_size)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.affine_do_ratio = affine_do_ratio
        self.device = device
        self.layer_n = 1

    def __call__(self, xs, ts, rate):
        os = self.predict(xs)
        ts2 = F.squeeze(ts)
        loss = F.softmax_cross_entropy(os, ts2)
        self.loss = loss
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, xs):
        os = self.predict(xs)
        os2 = F.sigmoid(os)
        return os2

    def predict(self, x):
        x2 = [item for item in x]
        if self.device >= 0:
            init = np.zeros((self.layer_n, len(x2), self.hidden_size), dtype=np.float32)
            hx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
            cx = Variable(chainer.cuda.to_gpu(init), volatile=not self.train)
        else:
            hx = None
            cx = None
        h = self.lstm1(hx, cx, x2, train=self.train)[2]
        h = [F.expand_dims(item, 0) for item in h]
        h = F.concat(h, 0)
        h = F.dropout(F.relu(self.l1(h)), train=self.train, ratio=self.affine_do_ratio)
        o = self.l2(h)
        return o