# QuickTorch

## Quickstart:

```python
import torch
import sys
from os import path
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

## Explaination

Each run of a model creates an output folder, and a project subdirectory. In the above example that would be `./ouput/MyModel`.

The model constructor supports the following parameters:

- layers `dict`: The tensor layers that are used in the model.
- loss `torch.nn (default = torch.nn.MSELoss)`: The loss to be initialized for the model.
- lr `double (default = .001)`: The learning rate for the model.
- decay_rate `double (default = .5)`: Decay rate for the exponential decay.
- decay_steps `int (default = 10000)`: Decay steps for the exponential decay.
- decay `bool (default = False)`: Defined if exponential decay should be used.
- weight_init `torch.nn.init (default = torch.nn.init.xavier_uniform_)`: Initializer for the layer tensors.
- optimizer `torch.optim (default = torch.optim.Adam)`: The backprop optimizer.
- batch_size `int (default = 32)`: Batch size used during training.
- accuracy `string (default = None)`: binary / categorical / None.

The `epoch` function trains the model on the given data:

- x `torch.Tensor`: Input values encoded as pytorch tensor.
- y `torch.Tensor`: Output values encoded as pytorch tensor.
- x_validation `torch.Tensor (default = [])`: Input validation values encoded as pytorch tensor.
- y_validation `torch.Tensor (default = [])`: Output validation values encoded as pytorch tensor.
- epochs `int`: Amount of epochs to run.
- save_best `str`: Save the best model according to values in _state.
- strict_batchsize `bool`: Whether batches with # of samples < batch_size can be used.

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

Quicktorch uses `tensorboardX` to show some training information. Output information is saved into the `./output/` directory. Tensorboard can then be started by using: `tensorboard --logdir ./output/[MyModel]`. The model class name is automatically used to create the logging directory.

