
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

            "linear1": torch.nn.Linear(13, 10),
            "batchnorm1": torch.nn.BatchNorm1d(10),
            "dropout1": torch.nn.Dropout(.4),

            "linear2": torch.nn.Linear(10, 8),
            "batchnorm2": torch.nn.BatchNorm1d(8),
            "dropout2": torch.nn.Dropout(.4),

            "linear3": torch.nn.Linear(8, 5),
            "batchnorm3": torch.nn.BatchNorm1d(5),
            "dropout3": torch.nn.Dropout(.4),

            "linear4": torch.nn.Linear(5, 1)

        }, batch_size=batch_size, lr=.001, decay=True)

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
