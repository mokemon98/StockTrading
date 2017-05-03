import math
import numpy as np

import chainer.functions as F
import chainer.links as L
from chainer import reporter, Chain, Variable


class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            # input_shape = (1, 10, 40)
            conv1_1 = L.Convolution2D(  1,  32,  (10, 3), stride=1, pad=0),
            conv1_2 = L.Convolution2D( 32,  64,  ( 1, 3), stride=1, pad=0),
            conv2_1 = L.Convolution2D( 64, 128,  ( 1, 3), stride=1, pad=0),
            conv2_2 = L.Convolution2D(128, 128,  ( 1, 1), stride=1, pad=0),
            conv3_1 = L.Convolution2D(128, 256,  ( 1, 3), stride=1, pad=0),
            conv3_2 = L.Convolution2D(256, 256,  ( 1, 1), stride=1, pad=0),
            conv4_1 = L.Convolution2D(256, 512,  ( 1, 3), stride=1, pad=0),
            conv4_2 = L.Convolution2D(512, 512,  ( 1, 1), stride=1, pad=0),
            l1 = L.Linear(512, 128),
            l2 = L.Linear(128, 1),
            l3 = L.Linear(128, 1),
            l4 = L.Linear(128, 1),

            ll1 = L.Linear(200, 100),
            ll2 = L.Linear(100, 25),
            ll3 = L.Linear(25, 1),
        )
        self.train = True

    def __call__(self, x, t1):
        #c, mu, var = self.predict(x)
        c = self.predict2(x)
        loss1 = F.sigmoid_cross_entropy(c, t1, use_cudnn=False)
        #loss2 = F.gaussian_nll(t2, mu, var)
        #self.loss = loss1 + loss2
        self.loss = loss1
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, x):
        #c, mu, var = self.predict(x)
        #return c, mu, var
        c = self.predict2(x)
        c2 = F.sigmoid(c)
        return c2

    def predict(self, x):
        h = F.relu(self.conv1_1(x))  # (32, 1, 38)
        h = F.relu(self.conv1_2(h))  # (64, 1, 36)
        h = F.max_pooling_2d(h, (1, 2), stride=(1, 2))  # (64, 1, 18)
        h = F.relu(self.conv2_1(h))  # (128, 1, 16)
        #h = F.relu(self.conv2_2(h))  # (128, 1, 16)
        h = F.max_pooling_2d(h, (1, 2), stride=(1, 2))  # (128, 1, 8)
        h = F.relu(self.conv3_1(h))  # (256, 1, 6)
        #h = F.relu(self.conv3_2(h))  # (256, 1, 6)
        h = F.max_pooling_2d(h, (1, 2), stride=(1, 2))  # (256, 1, 3)
        h = F.relu(self.conv4_1(h))  # (512, 1, 1)
        #h = F.relu(self.conv4_2(h))  # (512, 1, 1)
        #h = F.dropout(F.relu(self.l1(h)), train=self.train)
        h = F.relu(self.l1(h))
        o1 = self.l2(h)
        #o2 = self.l3(h)
        #o3 = self.l4(h)
        #return o1, o2, o3
        return o1

    def predict2(self, x):
        h = F.relu(self.ll1(x))
        h = F.relu(self.ll2(h))
        o = self.ll3(h)
        return o


class MyChainLSTM(Chain):

    def __init__(self, in_size, hidden_size, seq_size, device, lstm_do_ratio=0.0, affine_do_ratio=0.5):
        super(MyChainLSTM, self).__init__(
            lstm1 = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            l1 = L.Linear(hidden_size*seq_size, 50),
            l2 = L.Linear(50, 1)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.affine_do_ratio = affine_do_ratio
        self.device = device
        self.layer_n = 1

    def __call__(self, xs, ts):
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

    def __init__(self, in_size, hidden_size, seq_size, device, lstm_do_ratio=0.0, affine_do_ratio=0.5):
        super(MyChainBiDirectionalLSTM, self).__init__(
            lstm_f = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            lstm_b = L.NStepLSTM(1, in_size, hidden_size, lstm_do_ratio),
            l1 = L.Linear(hidden_size*seq_size*2, 50),
            l2 = L.Linear(50, 1)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.affine_do_ratio = affine_do_ratio
        self.device = device
        self.layer_n = 1

    def __call__(self, xs, ts):
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