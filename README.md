# QuickTorch

## PyTorch quickstart

```python
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import torch
import sys
from os import path
from sklearn.preprocessing import StandardScaler
from quicktorch.QuickTorch import QuickTorch

class MyModel(QuickTorch):

    # --------------------------------------------------------------------
    def __init__(self, batch_size):
        super(QuickTorchTest, self).__init__({
            "relu": torch.nn.ReLU()
        }, batch_size=64, lr=.001, decay=False)

    # --------------------------------------------------------------------
    def forward(self, input):
        X = input
        return X

model = MyModel()
model.visualize(torch.from_numpy(x_test).float())
model.epoch(x_train, y_train, x_test, y_test, epochs=1000)
model.saveModel("./output/logic.model")

```
