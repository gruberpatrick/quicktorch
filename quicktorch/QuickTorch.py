import torch
import torch.utils.data
import math
import numpy as np
from tensorboardX import SummaryWriter
import time
import os
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

try:
    from torchviz import make_dot
except Exception:
    logging.warning("Library 'torchviz' is not available. Model plotting not available.")
    pass


##########################################################################
class QuickTorch(torch.nn.Module):

    _name = ""
    _timestamp = str(int(time.time()))
    _tensors = {}
    _step = 0
    _batch_size = 32
    _lr = 0.001
    _decay_rate = 0.5
    _decay_steps = 10000
    _optimizer = None
    _decay = False
    _loss_epoch_history = []
    _loss_validation_history = []
    _graph_group_size = 10
    _writer = None
    _accuracy = None
    _stats = {
        "loss_epoch": [],
        "loss_batch": [],
        "loss_validation": [],
        "acc_epoch": [],
        "acc_batch": [],
        "acc_validation": [],
        "lr": [],
    }
    _layers = []
    _env = None
    _score = 0
    _state = None
    _device = None
    _path = None

    # --------------------------------------------------------------------
    def __init__(
        self,
        layers,
        loss=torch.nn.MSELoss,
        lr=0.001,
        decay_rate=0.5,
        decay_steps=10000,
        decay=False,
        weight_init=torch.nn.init.xavier_uniform_,
        optimizer=torch.optim.Adam,
        batch_size=32,
        accuracy=None,
        device=None,
        path=None,
    ):
        """
        Parameters
        ----------
        layers : dict
            The tensor layers that are used in the model
        loss : torch.nn (default = torch.nn.MSELoss)
            The loss to be initialized for the model
        lr : double (default = .001)
            The learning rate for the model
        decay_rate : double (default = .5)
            Decay rate for the exponential decay
        decay_steps : int (default = 10000)
            Decay steps for the exponential decay
        decay : bool (default = False)
            Defined if exponential decay should be used
        weight_init : torch.nn.init (default = torch.nn.init.xavier_uniform_)
            Initializer for the layer tensors
        optimizer : torch.optim (default = torch.optim.Adam)
            The backprop optimizer
        batch_size : int (default = 32)
            Batch size used during training
        accuracy : string (default = None)
            binary / categorical / None
        """

        # initialize tensors;
        super(QuickTorch, self).__init__()
        for it in layers:
            setattr(self, it, layers[it])
            self._layers.append(it)
        self.initializeWeights(weight_init, layers)

        # initialize decay and loss;
        self._batch_size = batch_size
        self._lr = lr
        self._decay = decay
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._optimizer = optimizer(self.parameters(), lr=self._lr)
        self._loss = loss()
        self._name = type(self).__name__
        self._accuracy = accuracy
        self._device = device
        self._path = path

        # overwrite the timestamp, to make it usable in a notebook;
        self._timestamp = str(int(time.time()))

    # --------------------------------------------------------------------
    def forward(self, input):
        """ Forward definition

        Forward propagation for the model. This function needs to be
        overwritten.

        Parameters
        ----------
        x : torch.Tensor
            Input values encoded as pytorch tensor

        Returns
        -------
        input : torch.Tensor
            Returns the modified tensor after forward propagation
        """

        logger.exception("[ERROR] Function 'forward' not implemented.")
        return input

    # --------------------------------------------------------------------
    def initializeWeights(self, weight_init, tensors):
        """ Initialize tensor weights

        Function loops through model parameters and initializes
        associated weights.

        Parameters
        ----------
        weight_init : torch.nn.init
            Weight initializer as given during initialization
        tensors : dict
            Tensors provided during initialization
        """

        for it in tensors:
            try:
                weight_init(tensors[it].weight)
            except Exception:
                continue

    # --------------------------------------------------------------------
    def exponentialDecay(self):
        """ Exponential decay implementation

        Adjust current learning rate by applying the exponential decay
        and adjusting the optimizer parameters.
        """

        lr = self._lr * pow(self._decay_rate, self._step / self._decay_steps)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    # -----------------------------------------------------------
    def train(self, x, y):
        """ Train execution for single batch

        Perform all training steps for a single minibatch. Calculate loss,
        perform backprop and adjust optimizer. If possible, calculates
        accuracy.

        Parameters
        ----------
        x : torch.Tensor
            Input values encoded as pytorch tensor
        y : torch.Tensor
            Output values encoded as pytorch tensor

        Returns
        -------
        loss : double
            Mean of minibatch losses
        accuracy : double
            Mean of minibatch accuracies
        y_hat : torch.Tensor
            Output values encoded as pytorch tensor
        """

        self._optimizer.zero_grad()

        # training step;
        y_hat = self(x)
        loss = self._loss(y_hat, y)

        loss.backward()
        self._optimizer.step()

        # decay;
        self._step += 1
        if self._decay:
            self.exponentialDecay()
        self._stats["lr"].append(np.array([param_group["lr"] for param_group in self._optimizer.param_groups]).mean())

        # calculate accuracy;
        if self._accuracy == "categorical":
            pred = y_hat.data.max(1)[1]
            correct = pred.eq(y.data).float().sum()
            accuracy = correct / len(y)
            return loss.item(), accuracy.item(), y_hat
        elif self._accuracy == "binary":
            pos = y_hat.data >= 0.5
            check = y.data >= 0.5
            correct = pos == check
            return loss.item(), correct.sum().float() / correct.size()[0], y_hat
        else:
            return loss.item(), -1, y_hat

    # -----------------------------------------------------------
    def test(self, x, y):
        """ Train execution for single batch

        Perform all evaluation steps for a single minibatch and
        calculate loss. If possible, calculates accuracy.

        Parameters
        ----------
        x : torch.Tensor
            Input values encoded as pytorch tensor
        y : torch.Tensor
            Output values encoded as pytorch tensor

        Returns
        -------
        loss : double
            Mean of minibatch losses
        accuracy : double
            Mean of minibatch accuracies
        y_hat : torch.Tensor
            Output values encoded as pytorch tensor
        """

        # calculate loss;
        y_hat = self(x)
        loss = self._loss(y_hat, y)

        # calculate accuracy;
        if self._accuracy == "categorical":
            pred = y_hat.data.max(1)[1]
            correct = pred.eq(y.data).float().sum()
            accuracy = correct / len(y)
            return loss.item(), accuracy.item(), y_hat
        elif self._accuracy == "binary":
            pos = y_hat.data >= 0.5
            check = y.data >= 0.5
            correct = pos == check
            return loss.item(), correct.sum().float() / correct.size()[0], y_hat
        else:
            return loss.item(), -1, y_hat

    # -----------------------------------------------------------
    def predict(self, x):
        """ Train execution for single batch

        Perform all evaluation steps for a single minibatch and
        calculate loss. If possible, calculates accuracy.

        Parameters
        ----------
        x : torch.Tensor
            Input values encoded as pytorch tensor

        Returns
        -------
        y_hat : torch.Tensor
            Output values encoded as pytorch tensor
        """

        return self(x)

    # -----------------------------------------------------------
    def visualize(self, x):
        """ Visualize network
        Exports a PDF displaying the neural network strcuture.
        """

        if not make_dot:
            return

        dot = make_dot(self(x), params=dict(self.named_parameters()))
        dot.render("./output/" + self._name + "/topology.gv", view=False)

    # -----------------------------------------------------------
    def showNNStats(self, attributes=["weight", "bias"]):
        """ Visualize network layer weights

        Users tensorflow to display some of the basic NN structure
        information to better understand weight distribution.
        """

        for layer in self._layers:
            for att in attributes:
                try:
                    self._writer.add_histogram(layer + "_weight", getattr(getattr(self, layer), att))
                except Exception:
                    pass

    # -----------------------------------------------------------
    def saveModel(self, path=None):
        """ Save model

        Save the current model to the given path.

        Parameters
        ----------
        path : str
            The path to where the model will be saved
        """

        state_dict = {"_state_dict": self.state_dict()}
        state_dict["_step"] = self._step
        state_dict["_batch_size"] = self._batch_size
        state_dict["_lr"] = self._lr
        state_dict["_decay_rate"] = self._decay_rate
        state_dict["_decay_steps"] = self._decay_steps
        state_dict["_optimizer"] = self._optimizer
        state_dict["_decay"] = self._decay
        state_dict["_stats"] = self._stats

        final_path = "./output/" + self._name + "/" + self._timestamp + ".model"

        if self._path:
            final_path = os.path.join(self._path, self._name + "/" + self._timestamp + ".model")
        elif path:
            final_path = os.path.join(path, self._name + "/" + self._timestamp + ".model")

        torch.save(state_dict, final_path)

    # -----------------------------------------------------------
    def loadModel(self, path):
        """ Load model

        Load model from the given path.

        Parameters
        ----------
        path : str
            The path from where the model will be loaded
        """

        state_dict = torch.load(path, map_location=self._device if self._device else torch.device("cpu"))
        self._step = state_dict["_step"]
        self._batch_size = state_dict["_batch_size"]
        self._lr = state_dict["_lr"]
        self._decay_rate = state_dict["_decay_rate"]
        self._decay_steps = state_dict["_decay_steps"]
        self._optimizer = state_dict["_optimizer"]
        self._decay = state_dict["_decay"]
        self._stats = state_dict["_stats"]
        self.load_state_dict(state_dict["_state_dict"])

    # -----------------------------------------------------------
    def createProjectFolder(self):
        """ Create a new project folder

        Creates a timestamped folder and initializes the SummaryWriter.
        """

        try:
            os.mkdir("./output/" + self._name + "/")
        except Exception:
            pass

        path = "./output/" + self._name + "/" + self._timestamp + "_tb/"
        if self._path:
            path = os.path.join(self._path, self._name + "/" + self._timestamp + "_tb/")

        logger.debug("Creating writer: \"{}\"".format(path))
        self._writer = SummaryWriter(log_dir=path)

    # -----------------------------------------------------------
    def getBatches(self, x, y, strict_batchsize):

        amount = math.ceil(x.shape[0] / self._batch_size)
        batches = list(range(amount))
        np.random.shuffle(batches)
        minibatch_count = 0

        for it in batches:

            minibatch_count += 1

            if (
                strict_batchsize
                and x[(it) * self._batch_size: (it + 1) * self._batch_size].shape[0] != self._batch_size
            ):
                continue

            yield minibatch_count, (
                torch.from_numpy(x[(it) * self._batch_size: (it + 1) * self._batch_size]),
                torch.from_numpy(y[(it) * self._batch_size: (it + 1) * self._batch_size]),
            )

    # -----------------------------------------------------------
    def epoch(self, x, y=None, x_validation=[], y_validation=[], epochs=1, save_best="", strict_batchsize=False):
        """ Train execution for entire data set

        Create minibatchces and call train function. Keeps track of
        historic statistical values to analyze performance.

        Parameters
        ----------
        x : torch.Tensor
            Input values encoded as pytorch tensor
        y : torch.Tensor
            Output values encoded as pytorch tensor
        x_validation : torch.Tensor (default = [])
            Input validation values encoded as pytorch tensor
        y_validation : torch.Tensor (default = [])
            Output validation values encoded as pytorch tensor
        epochs : int
            Amount of epochs to run
        save_best : str
            Save the best model according to values in _state
        strict_batchsize : bool
            Whether batches with # of samples < batch_size can be used
        """

        self.createProjectFolder()

        best = {"acc": -np.inf, "loss": np.inf, "acc_validation": -np.inf, "loss_validation": np.inf, "trigger": []}
        self._epoch = 0
        self._total_epochs = epochs
        for epoch in range(self._epoch, self._total_epochs):

            best["trigger"] = []

            self._epoch = epoch

            if isinstance(x, torch.utils.data.DataLoader) and not y:
                batches = enumerate(x, 1)
                amount = math.ceil(len(x.sampler) / self._batch_size)
            else:
                batches = self.getBatches(x, y, strict_batchsize)
                amount = math.ceil(x.shape[0] / self._batch_size)

            sum_loss = 0
            sum_acc = 0

            for minibatch_count, (x_train, y_train) in batches:

                if self._device:
                    x_train, y_train = x_train.to(self._device), y_train.to(self._device)

                loss, acc, _ = self.train(x_train, y_train)

                self._stats["loss_batch"].append(loss)
                self._stats["acc_batch"].append(acc)
                self._writer.add_scalar(
                    self._name + "/loss_batch", self._stats["loss_batch"][-1], (epoch * amount) + minibatch_count
                )
                self._writer.add_scalar(
                    self._name + "/acc_batch", self._stats["acc_batch"][-1], (epoch * amount) + minibatch_count
                )

                sum_loss += loss
                sum_acc += acc

                logger.debug(
                    "[%5d / %5d] Batch: %5d of %5d - loss: %8.4f, acc: %8.4f"
                    % (
                        epoch + 1,
                        epochs,
                        minibatch_count,
                        amount,
                        self._stats["loss_batch"][-1],
                        self._stats["acc_batch"][-1] * 100,
                    )
                )

            # self._loss_epoch_history.append(sum_loss / amount)
            self._stats["loss_epoch"].append(sum_loss / amount)
            self._stats["acc_epoch"].append(sum_acc / amount)
            self._writer.add_scalar(self._name + "/loss_epoch", self._stats["loss_epoch"][-1], epoch)
            self._writer.add_scalar(self._name + "/acc_epoch", self._stats["acc_epoch"][-1], epoch)
            self._writer.add_scalar(self._name + "/lr", self._stats["lr"][-1], epoch)

            # save the current best scores;
            if self._score > best["acc"]:
                best["acc"] = self._stats["acc_epoch"][-1]
                best["trigger"].append("acc")
            if self._step < best["loss"]:
                best["loss"] = self._stats["loss_epoch"][-1]
                best["trigger"].append("loss")

            if x_validation != [] and x_validation is not None:

                validation_loss = []
                validation_acc = []
                batches = (
                    enumerate(x_validation, 1)
                    if isinstance(x_validation, torch.utils.data.DataLoader) and not y_validation else
                    self.getBatches(x_validation, y_validation, strict_batchsize)
                )

                for minibatch_count, (x_val, y_val) in batches:

                    if self._device:
                        x_val, y_val = x_val.to(self._device), y_val.to(self._device)

                    loss, acc, _ = self.test(x_val, y_val)
                    validation_loss.append(loss)
                    validation_acc.append(acc)

                self._stats["loss_validation"].append(np.array(validation_loss).mean())
                self._stats["acc_validation"].append(np.array(validation_acc).mean())
                self._writer.add_scalar(self._name + "/loss_validation", self._stats["loss_validation"][-1], epoch)
                self._writer.add_scalar(self._name + "/acc_validation", self._stats["acc_validation"][-1], epoch)

                # save the current best validation scores;
                if self._step < best["loss_validation"]:
                    best["loss_validation"] = self._stats["loss_validation"][-1]
                    best["trigger"].append("loss_validation")
                if self._step > best["acc_validation"]:
                    best["acc_validation"] = self._stats["acc_validation"][-1]
                    best["trigger"].append("acc_validation")

            if save_best != "" and save_best in best["trigger"]:
                self.saveModel()
                logger.debug("New model saved...")

            logger.debug(
                "[%5d / %5d] loss: %8.4f, val_loss: %8.4f\tacc: %8.4f, val_acc: %8.4f"
                % (
                    epoch + 1,
                    epochs,
                    self._stats["loss_epoch"][-1],
                    self._stats["loss_validation"][-1] if len(self._stats["loss_validation"]) > 0 else -1,
                    self._stats["acc_epoch"][-1] * 100,
                    self._stats["acc_validation"][-1] * 100 if len(self._stats["acc_validation"]) > 0 else -1,
                )
            )
