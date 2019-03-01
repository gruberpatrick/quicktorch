
import torch
import unittest
import sys
from os import path
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from quicktorch.QuickTorch import QuickTorch

##########################################################################
class QuickTorchTest(QuickTorch):

    # --------------------------------------------------------------------
    def __init__(self, batch_size):

        super(QuickTorchTest, self).__init__({

            "relu": torch.nn.ReLU(),

            "linear1": torch.nn.Linear(13, 30),
            "batchnorm1": torch.nn.BatchNorm1d(30),
            "dropout1": torch.nn.Dropout(.4),

            "linear2": torch.nn.Linear(30, 25),
            "batchnorm2": torch.nn.BatchNorm1d(25),
            "dropout2": torch.nn.Dropout(.4),

            "linear3": torch.nn.Linear(25, 20),
            "batchnorm3": torch.nn.BatchNorm1d(20),
            "dropout3": torch.nn.Dropout(.4),

            "linear4": torch.nn.Linear(20, 15),
            "batchnorm4": torch.nn.BatchNorm1d(15),
            "dropout4": torch.nn.Dropout(.4),

            "linear5": torch.nn.Linear(15, 8),
            "batchnorm5": torch.nn.BatchNorm1d(8),
            "dropout5": torch.nn.Dropout(.4),

            "linear6": torch.nn.Linear(8, 5),
            "batchnorm6": torch.nn.BatchNorm1d(5),
            "dropout6": torch.nn.Dropout(.4),

            "linear_side1": torch.nn.Linear(30, 29),
            "batchnorm_side1": torch.nn.BatchNorm1d(29),
            "dropout_side1": torch.nn.Dropout(.4),

            "linear_side2": torch.nn.Linear(29, 28),
            "batchnorm_side2": torch.nn.BatchNorm1d(28),
            "dropout_side2": torch.nn.Dropout(.4),

            "linear_side3": torch.nn.Linear(28, 27),
            "batchnorm_side3": torch.nn.BatchNorm1d(27),
            "dropout_side3": torch.nn.Dropout(.4),

            "linear_side4": torch.nn.Linear(27, 25),
            "batchnorm_side4": torch.nn.BatchNorm1d(25),
            "dropout_side4": torch.nn.Dropout(.4),

            "linear7": torch.nn.Linear(5, 1)

        }, batch_size=batch_size, lr=.001, decay=False)

    # --------------------------------------------------------------------
    def sidebar(self, X):

        X = self.linear_side1(X)
        X = self.relu(X)
        X = self.batchnorm_side1(X)
        X = self.dropout_side1(X)

        X = self.linear_side2(X)
        X = self.relu(X)
        X = self.batchnorm_side2(X)
        X = self.dropout_side2(X)

        X = self.linear_side3(X)
        X = self.relu(X)
        X = self.batchnorm_side3(X)
        X = self.dropout_side3(X)

        X = self.linear_side4(X)
        X = self.relu(X)
        X = self.batchnorm_side4(X)
        X = self.dropout_side4(X)

        return X

    # --------------------------------------------------------------------
    def forward(self, input):

        X = self.linear1(input)
        X = self.relu(X)
        X = self.batchnorm1(X)
        X = self.dropout1(X) 

        X_pre = self.sidebar(X)

        X = self.linear2(X)
        X = self.relu(X)
        X = self.batchnorm2(X)
        X = self.dropout2(X)

        X = X + X_pre

        X = self.linear3(X)
        X = self.relu(X)
        X = self.batchnorm3(X)
        X = self.dropout3(X)
        
        X = self.linear4(X)
        X = self.relu(X)
        X = self.batchnorm4(X)
        X = self.dropout4(X)

        X = self.linear5(X)
        X = self.relu(X)
        X = self.batchnorm5(X)
        X = self.dropout5(X)

        X = self.linear6(X)
        X = self.relu(X)
        X = self.batchnorm6(X)
        X = self.dropout6(X)
        
        X = self.linear7(X)

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
        qt = QuickTorchTest(64)
        qt.visualize(torch.from_numpy(x_test).float())
        qt.epoch(x_train, y_train, x_test, y_test, epochs=2000)
        qt.saveModel()

        # analyze output;
        y_hat = qt.predict(torch.from_numpy(x_test).float()).tolist()
        plot([go.Scatter(
                x = list(range(len(y_test.tolist()))),
                y = [it[0] for it in out_scaler.inverse_transform(y_test)],
                mode = "markers",
                name = "Actual"
            ), go.Scatter(
                x = list(range(len(y_hat))),
                y = [it[0] for it in out_scaler.inverse_transform(y_hat)],
                mode = "markers",
                name = "Pred"
            )], filename="./output/test.html", auto_open=False)

        self.assertLessEqual(qt._stats["loss_epoch"][-1], .5)
        print("  ", qt._stats["loss_epoch"][-1], .5)

################################################################################
if __name__ == '__main__':

    unittest.main()
