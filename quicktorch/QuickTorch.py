
import torch
import math
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from torchviz import make_dot
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import os
from tensorboard import default, program

##########################################################################
class QuickTorch(torch.nn.Module):

    _name = ""
    _timestamp = str(int(time.time()))
    _tensors = {}
    _step = 0
    _batch_size = 32
    _lr = .001
    _decay_rate = .5
    _decay_steps = 10000
    _optimizer = None
    _decay = False
    _loss_epoch_history = []
    _loss_batch_history = []
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
        "lr": []
    }
    _layers = []

    # --------------------------------------------------------------------
    def __init__(self, tensors, loss=torch.nn.MSELoss, lr=.001, decay_rate=.5, 
            decay_steps=10000, decay=False, weight_init=torch.nn.init.xavier_uniform_, 
            optimizer=torch.optim.Adam, batch_size=32, accuracy=None):
        """
        Parameters
        ----------
        tensors : dict
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
        accuracy : string
            binary / categorical / None
        """

        # initialize tensors;
        super(QuickTorch, self).__init__()
        for it in tensors:
            setattr(self, it, tensors[it])
            self._layers.append(it)
        self.initializeWeights(weight_init, tensors)

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

        # create project folder;
        try: os.mkdir("./output/" + self._name + "/")
        except: pass
        self._writer = SummaryWriter(log_dir="./output/" + self._name + "/" + self._timestamp + "_tb/")
        
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

        print("[ERROR] Not implemented.")
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
            try: weight_init(tensors[it].weight)
            except: continue

    # --------------------------------------------------------------------
    def exponentialDecay(self):
        """ Exponential decay implementation

        Adjust current learning rate by applying the exponential decay
        and adjusting the optimizer parameters.
        """
        
        lr = self._lr * pow(self._decay_rate, self._step / self._decay_steps)
        for param_group in self._optimizer.param_groups: param_group["lr"] = lr

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
        
        # training step;
        y_hat = self(x)
        loss = self._loss(y_hat, y)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # decay;
        self._step += 1
        if self._decay: self.exponentialDecay()
        self._stats["lr"].append(np.array([param_group["lr"] for param_group in self._optimizer.param_groups]).mean())

        # calculate accuracy;
        if self._accuracy == "categorical":
            pred = y_hat.data.max(1)[1]
            correct = pred.eq(y.data).float().sum()
            accuracy = correct / len(y)
            return loss.item(), accuracy.item(), y_hat
        elif self._accuracy == "binary":
            pos = y_hat.data >= .5
            check = y.data >= .5
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
            pos = y_hat.data >= .5
            check = y.data >= .5
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

        dot = make_dot(self(x), params=dict(self.named_parameters()))
        dot.render("./output/" + self._name + "/topology.gv", view=False)  

    # -----------------------------------------------------------
    def epoch(self, x, y, x_validation=[], y_validation=[], epochs=1, save_best="", strict_batchsize=False):
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
        """

        self._epoch = 0
        self._total_epochs = epochs
        for epoch in range(self._epoch, self._total_epochs):

            self._epoch = epoch
            self._loss_batch_history = []
            amount = math.ceil(x.shape[0] / self._batch_size)
            batches = list(range(amount))
            np.random.shuffle(batches)
            sum_loss = 0
            sum_acc = 0
            minibatch_count = 0
            for it in batches:

                if strict_batchsize and len(x[ (it)*self._batch_size : (it+1)*self._batch_size ]) < self._batch_size:
                    continue

                data = [
                    torch.from_numpy(x[ (it)*self._batch_size : (it+1)*self._batch_size ]),
                    torch.from_numpy(y[ (it)*self._batch_size : (it+1)*self._batch_size ])
                ]
                loss, acc, _ = self.train(data[0], data[1])
                
                #self._loss_batch_history.append(loss)
                self._stats["loss_batch"].append(loss)
                self._stats["acc_batch"].append(acc)
                self._writer.add_scalar(self._name + "/loss_batch", self._stats["loss_batch"][-1], (epoch * amount) + minibatch_count)
                self._writer.add_scalar(self._name + "/acc_batch", self._stats["acc_batch"][-1], (epoch * amount) + minibatch_count)

                sum_loss += loss
                sum_acc += acc
                minibatch_count += 1

                print("\t\r[%5d / %5d] Batch: %5d of %5d - loss: %8.4f, acc: %8.4f" % \
                    (epoch+1, epochs, minibatch_count, amount, self._stats["loss_batch"][-1], self._stats["acc_batch"][-1]*100), end="")

            #self._loss_epoch_history.append(sum_loss / amount)
            self._stats["loss_epoch"].append(sum_loss / amount)
            self._stats["acc_epoch"].append(sum_acc / amount)
            self._writer.add_scalar(self._name + "/loss_epoch", self._stats["loss_epoch"][-1], epoch)
            self._writer.add_scalar(self._name + "/acc_epoch", self._stats["acc_epoch"][-1], epoch)
            self._writer.add_scalar(self._name + "/lr", self._stats["lr"][-1], epoch)

            validation_loss = []
            validation_acc = []
            for it in range(math.ceil(x_validation.shape[0] / self._batch_size)):

                data = [
                    torch.from_numpy(x_validation[ (it)*self._batch_size : (it+1)*self._batch_size ]),
                    torch.from_numpy(y_validation[ (it)*self._batch_size : (it+1)*self._batch_size ])
                ]
                loss, acc, _ = self.test(data[0], data[1])
                validation_loss.append(loss)
                validation_acc.append(acc)

            #self._loss_validation_history.append(loss)
            self._stats["loss_validation"].append(np.array(validation_loss).mean())
            self._stats["acc_validation"].append(np.array(validation_acc).mean())
            self._writer.add_scalar(self._name + "/loss_validation", self._stats["loss_validation"][-1], epoch)
            self._writer.add_scalar(self._name + "/acc_validation", self._stats["acc_validation"][-1], epoch)

            print("\t\r[%5d / %5d] loss: %8.4f, val_loss: %8.4f\tacc: %8.4f, val_acc: %8.4f" % 
                (epoch+1, epochs, (sum_loss/minibatch_count), loss, self._stats["acc_epoch"][-1]*100, self._stats["acc_validation"][-1]*100))

    # -----------------------------------------------------------
    def showNNStats(self, attributes=["weight", "bias"]):
        """ Visualize network layer weights

        Users tensorflow to display some of the basic NN structure
        information to better understand weight distribution.
        """

        for layer in self._layers:
            for att in attributes:
                try: self._writer.add_histogram(layer + "_weight", getattr(getattr(self, layer), att))
                except: pass

    # -----------------------------------------------------------
    def exportStats(self):
        """ Visualize network training performance

        Exports a HTML plotly file that allows exploration of
        network performance.
        """

        xs = []
        grouped_epoch_history = []
        grouped_validation_history = []

        for it in range(0, len(self._stats["loss_epoch"]), self._graph_group_size):
            grouped_epoch_history.append( np.mean(np.array(self._stats["loss_epoch"][it:it+self._graph_group_size])).tolist() )
            grouped_validation_history.append( np.mean(np.array(self._stats["loss_validation"][it:it+self._graph_group_size])).tolist() )
            xs.append(it + (self._graph_group_size / 2))

        plot([go.Scatter(
            x = list(range(len(self._stats["loss_epoch"]))),
            y = self._stats["loss_epoch"],
            name = "Epoch Training Loss"
        ), go.Scatter(
            x = list(range(len(self._stats["loss_validation"]))),
            y = self._stats["loss_validation"],
            name = "Epoch Validation Loss"
        ), go.Scatter(
            x = xs,
            y = grouped_epoch_history,
            name = "Grouped Training Loss"
        ), go.Scatter(
            x = xs,
            y = grouped_validation_history,
            name = "Grouped Validation Loss"
        )], filename="./output/training.html", auto_open=False)

    # -----------------------------------------------------------
    def saveModel(self):
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
        torch.save(state_dict, "./output/" + self._name + "/" + self._timestamp + ".model")

    # -----------------------------------------------------------
    def loadModel(self, path):
        """ Load model

        Load model from the given path.

        Parameters
        ----------
        path : str
            The path from where the model will be loaded 
        """

        state_dict = torch.load(path)
        self._step = state_dict["_step"]
        self._batch_size = state_dict["_batch_size"]
        self._lr = state_dict["_lr"]
        self._decay_rate = state_dict["_decay_rate"]
        self._decay_steps = state_dict["_decay_steps"]
        self._optimizer = state_dict["_optimizer"]
        self._decay = state_dict["_decay"]
        self._stats = state_dict["_stats"]
        self.load_state_dict(state_dict["_state_dict"])

