import torch
import unittest
import sys
from os import path
import numpy as np
import pandas as pd

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from quicktorch.QuickTorch import QuickTorch
from quicktorch.Utils import Utils

##########################################################################
class ReviewLSTMTest(QuickTorch):

    _hidden = (torch.zeros(1, 64, 128), torch.zeros(1, 64, 128))
    _tag = None

    # --------------------------------------------------------------------
    def __init__(self, sentence_length, embedding_size):

        super(ReviewLSTMTest, self).__init__(
            {
                "relu": torch.nn.ReLU(),
                "embed": torch.nn.Embedding(embedding_size, 128),
                "lstm_layer1": torch.nn.LSTM(128, 128, bidirectional=True, batch_first=True, num_layers=2),
                "analyze1": torch.nn.Linear(257, 100),
                "analyze2": torch.nn.Linear(100, 50),
                "analyze3": torch.nn.Linear(50, 25),
                "output": torch.nn.Linear(25, 2),
                "softmax": torch.nn.LogSoftmax(dim=1),
                "dropout": torch.nn.Dropout(0.4),
            },
            batch_size=64,
            lr=0.001,
            decay=True,
            loss=torch.nn.NLLLoss,
            accuracy="categorical",
        )

        self._tag = torch.from_numpy((np.array(list(range(128)), dtype=np.float) / 127 * 2) - 1).float()
        self._tag = self._tag.reshape(128, 1)
        self._quick_tag = self._tag.expand(64, 128, 1)

    # --------------------------------------------------------------------
    def analyze(self, X):

        X = self.analyze1(X)
        X = self.relu(X)
        X = self.analyze2(X)
        # X = self.dropout(X)
        X = self.relu(X)
        X = self.analyze3(X)
        X = self.relu(X)

        return X

    # --------------------------------------------------------------------
    def forward(self, input):

        X = self.embed(input)
        X, self._hidden = self.lstm_layer1(X)

        # print(X.shape, self._tag.shape)
        # torch.Size([64, 128, 256]) torch.Size([128, 1])

        if X.shape[0] == 64:
            X_in = torch.cat([X, self._quick_tag], 2)
            shape = X_in.shape
        else:
            tag = self._tag.expand(X.shape[0], self._quick_tag.shape[1], self._quick_tag.shape[2])
            X_in = torch.cat([X, tag], 2)
            shape = X_in.shape

        # print(X_in.shape)
        # torch.Size([64, 128, 257])
        # print(X_in[0, 0, -2], X_in[0, 0, -1])
        # print(X_in[0, 1, -2], X_in[0, 1, -1])
        # print(X_in[0, 2, -2], X_in[0, 2, -1])

        X_in = X_in.reshape(shape[0] * shape[1], shape[2])

        # print(X_in.shape)
        X = self.analyze(X_in)
        # print(X.shape)
        X = X.reshape(shape[0], shape[1], X.shape[1])
        # print(X.shape)

        X = X.sum(1)
        # print(X.shape)

        X = self.output(X)
        X = self.softmax(X)

        return X


##########################################################################
class LSTMTest(unittest.TestCase):

    # --------------------------------------------------------------------
    def testModel(self):

        print("=======================================\n  testModel")

        data = pd.read_csv(
            "./testcases/amazon_reviews_mobile.tsv", sep='\t', error_bad_lines=False, skip_blank_lines=True
        )
        text = data["review_body"]
        labels = data["star_rating"]
        y = []
        for label in labels:
            if label >= 2.5:
                y.append(1)
            else:
                y.append(0)
        # y = y.reshape((y.shape[0], 1))
        x, word_idx, idx_word, counter, longest = Utils.indexColumn(text)
        x_train = np.array(Utils.padding(x[:13000], 128), dtype=np.int64)
        y_train = np.array(y[:13000], dtype=np.int64)
        x_test = np.array(Utils.padding(x[13000:14900], 128), dtype=np.int64)
        y_test = np.array(y[13000:14900], dtype=np.int64)

        # NN handler;
        qt = ReviewLSTMTest(128, len(list(word_idx.keys())))
        qt.epoch(x_train, y_train, x_test, y_test, epochs=5, strict_batchsize=True)
        qt.saveModel()

        self.assertGreaterEqual(qt._stats["acc_epoch"][-1], 0.8)


################################################################################
if __name__ == '__main__':

    unittest.main()
