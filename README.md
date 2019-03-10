# QuickTorch

## PyTorch Quickstart:

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

## Test Cases:

The test cases also give some information about how to use quicktorch. The options are:

#### Simple Feed Forward Regression:

  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/BostonHousingSimpleTest.py
  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/BostonHousingComplexTest.py

### Simple Feed Forward Classification:

  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/LeukemiaBinaryTest.py
  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/LeukemiaCategoricalTest.py

### RNN Classification:

  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/ReviewLSTMTest.py

### Deep Q Reinforcement Learning:

  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/CartPoleTest.py
  - https://github.com/gruberpatrick/quicktorch/blob/master/testcases/LunarLanderTest.py

## Tensorboard

Quicktorch uses `tensorboardX` to show some training information. Output information is saved into the `./output/` directory. Tensorboard can then be started by using: `tensorboard --logdir ./output/[model]/`. The model class name is automatically used to create the logging directory.

