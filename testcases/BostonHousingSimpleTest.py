import torch
import unittest
import sys
from os import path
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from quicktorch.QuickTorch import QuickTorch

##########################################################################
class BostonHousingSimpleTest(QuickTorch):

    # --------------------------------------------------------------------
    def __init__(self, batch_size):

        super(BostonHousingSimpleTest, self).__init__(
            {
                "relu": torch.nn.ReLU(),
                "linear1": torch.nn.Linear(13, 10),
                "batchnorm1": torch.nn.BatchNorm1d(10),
                "dropout1": torch.nn.Dropout(0.4),
                "linear2": torch.nn.Linear(10, 8),
                "batchnorm2": torch.nn.BatchNorm1d(8),
                "dropout2": torch.nn.Dropout(0.4),
                "linear3": torch.nn.Linear(8, 5),
                "batchnorm3": torch.nn.BatchNorm1d(5),
                "dropout3": torch.nn.Dropout(0.4),
                "linear4": torch.nn.Linear(5, 1),
            },
            batch_size=batch_size,
            lr=0.001,
            decay=False,
            decay_rate=0.1,
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
        X = self.dropout2(X)

        X = self.linear3(X)
        X = self.relu(X)
        X = self.batchnorm3(X)
        X = self.dropout3(X)

        X = self.linear4(X)

        return X


##########################################################################
class BostonHousingTest(unittest.TestCase):

    # --------------------------------------------------------------------
    def testModel(self):

        print("=======================================\n  testModel")

        # data preparation;
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        in_scaler = StandardScaler()
        out_scaler = StandardScaler()
        in_scaler.fit(x_train)
        x_train = in_scaler.transform(x_train)
        x_test = in_scaler.transform(x_test)
        y_train = y_train.reshape((x_train.shape[0], 1))
        y_test = y_test.reshape((x_test.shape[0], 1))
        out_scaler.fit(y_train)
        y_train = out_scaler.transform(y_train)
        y_test = out_scaler.transform(y_test)

        # NN handler;
        qt = BostonHousingSimpleTest(64)
        # qt.visualize(torch.from_numpy(x_test).float())
        qt.epoch(
            np.array(x_train, dtype=np.float32),
            np.array(y_train, dtype=np.float32),
            np.array(x_test, dtype=np.float32),
            np.array(y_test, dtype=np.float32),
            epochs=500,
        )
        qt.saveModel()
        qt.showNNStats()

        self.assertLessEqual(qt._stats["loss_epoch"][-1], 0.5)
        print("  ", qt._stats["loss_epoch"][-1], 0.5)


################################################################################
if __name__ == '__main__':

    unittest.main()
