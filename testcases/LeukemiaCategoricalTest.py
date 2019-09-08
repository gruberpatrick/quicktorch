import torch
import unittest
import sys
from os import path
import numpy as np

import h5py

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from quicktorch.QuickTorch import QuickTorch

##########################################################################
class LeukemiaCategoricalTest(QuickTorch):

    # --------------------------------------------------------------------
    def __init__(self, batch_size):

        super(LeukemiaCategoricalTest, self).__init__(
            {
                "relu": torch.nn.ReLU(),
                "linear1": torch.nn.Linear(7129, 5000),
                "batchnorm1": torch.nn.BatchNorm1d(5000),
                "dropout1": torch.nn.Dropout(0.6),
                "linear2": torch.nn.Linear(5000, 2048),
                "batchnorm2": torch.nn.BatchNorm1d(2048),
                "dropout2": torch.nn.Dropout(0.6),
                "linear3": torch.nn.Linear(2048, 1024),
                "batchnorm3": torch.nn.BatchNorm1d(1024),
                "dropout3": torch.nn.Dropout(0.6),
                "linear4": torch.nn.Linear(1024, 512),
                "batchnorm4": torch.nn.BatchNorm1d(512),
                "dropout4": torch.nn.Dropout(0.6),
                "linear5": torch.nn.Linear(512, 256),
                "batchnorm5": torch.nn.BatchNorm1d(256),
                "dropout5": torch.nn.Dropout(0.6),
                "linear6": torch.nn.Linear(256, 128),
                "batchnorm6": torch.nn.BatchNorm1d(128),
                "dropout6": torch.nn.Dropout(0.6),
                "output": torch.nn.Linear(128, 2),
                "softmax": torch.nn.LogSoftmax(dim=1),
            },
            batch_size=batch_size,
            lr=0.001,
            decay=False,
            loss=torch.nn.NLLLoss,
            accuracy="categorical",
        )

    # --------------------------------------------------------------------
    def forward(self, input):

        X = self.linear1(input)
        X = self.relu(X)
        X = self.batchnorm1(X)
        X = self.dropout1(X)

        X = self.linear2(X)
        X = self.relu(X)
        X = self.batchnorm2(X)
        # X = self.dropout2(X)

        X = self.linear3(X)
        X = self.relu(X)
        X = self.batchnorm3(X)
        X = self.dropout3(X)

        X = self.linear4(X)
        X = self.relu(X)
        X = self.batchnorm4(X)
        # X = self.dropout4(X)

        X = self.linear5(X)
        X = self.relu(X)
        X = self.batchnorm5(X)
        X = self.dropout5(X)

        X = self.linear6(X)
        X = self.relu(X)
        X = self.batchnorm6(X)
        # X = self.dropout6(X)

        X = self.output(X)
        X = self.softmax(X)

        return X


##########################################################################
class LeukemiaTest(unittest.TestCase):

    # --------------------------------------------------------------------
    def testModel(self):

        print("=======================================\n  testModel")

        fh = h5py.File("./testcases/leukemia.hdf5", "r")
        x = np.array(fh["features"], dtype=np.float32)
        y = np.array(fh["predictions"], dtype=np.int64)
        x_train = x[:29]
        y_train = y[:29]
        x_test = x[29:]
        y_test = y[29:]

        # NN handler;
        qt = LeukemiaCategoricalTest(64)
        qt.visualize(torch.from_numpy(x_test).float())
        qt.epoch(x_train, y_train, x_test, y_test, epochs=40)
        qt.saveModel()

        self.assertGreaterEqual(qt._stats["acc_epoch"][-1], 0.8)


################################################################################
if __name__ == '__main__':

    unittest.main()
