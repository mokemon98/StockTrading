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

    def __init__(self, in_size, hidden_size, seq_size):
        super(MyChainLSTM, self).__init__(
            lstm1 = L.LSTM(in_size, hidden_size),
            l1 = L.Linear(hidden_size*seq_size, 50),
            l2 = L.Linear(50, 1)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size

    def __call__(self, x, t1):
        c = self.predict(x)
        loss1 = F.sigmoid_cross_entropy(c, t1, use_cudnn=False)
        self.loss = loss1
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, x):
        c = self.predict(x)
        c2 = F.sigmoid(c)
        return c2

    def predict(self, x):
        batch_size = x.shape[0]
        for i in range(batch_size):
            self.lstm1.reset_state()
            tmp = F.expand_dims(self.lstm1(x[i]), 0)
            if i == 0:
                h = tmp
            else:
                h = F.vstack([h, tmp])
        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        o = self.l2(h)
        return o


class MyChainLSTM2(Chain):

    def __init__(self, in_size, hidden_size, seq_size):
        super(MyChainLSTM2, self).__init__(
            # input_shape = (1, 10, 40)
            lstm1 = L.NStepLSTM(1, in_size, hidden_size, 0.5),
            l1 = L.Linear(hidden_size*seq_size, 1)
        )
        self.train = True
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size

    def __call__(self, x, t1):
        c = self.predict(x)
        loss1 = F.sigmoid_cross_entropy(c, t1, use_cudnn=False)
        self.loss = loss1
        if math.isnan(self.loss.data):
            raise RuntimeError("Error in MyChain: loss.data is nan!")
        reporter.report({"loss": self.loss}, self)
        return self.loss

    def forward(self, x):
        c = self.predict(x)
        c2 = F.sigmoid(c)
        return c2

    def predict(self, x):
        from IPython import embed; embed()
        x2 = [item for item in x]
        h = self.lstm1(None, None, x2, train=self.train)
        o = [self.l1(item) for item in h]
        return o
