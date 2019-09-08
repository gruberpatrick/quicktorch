import torch
import unittest
import sys
from os import path
import numpy as np
import pandas as pd
import gym
from collections import deque
import random

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from quicktorch.QuickTorch import QuickTorch
from quicktorch.Utils import Utils

##########################################################################
class LunarLander(QuickTorch):

    _epsilon = 1.0
    _epsilon_min = 0.01
    _epsilon_decay = 0.998
    _mem = deque(maxlen=2000)
    _hist = []
    _state = []
    _gamma = 0.9
    _best_score = -np.inf

    # --------------------------------------------------------------------
    def __init__(self):

        self._env = gym.make("LunarLander-v2")
        self._action_size = self._env.action_space.n
        self._state_size = self._env.observation_space.shape[0]

        super(LunarLander, self).__init__(
            {
                "relu": torch.nn.ReLU(),
                "fc1": torch.nn.Linear(self._state_size, 32),
                "fc2": torch.nn.Linear(32, 64),
                "fc3": torch.nn.Linear(64, 32),
                "fc4": torch.nn.Linear(32, self._action_size),
                "relu": torch.nn.ReLU(),
                "softmax": torch.nn.Softmax(dim=1),
            },
            lr=0.0025,
            decay=False,
            loss=torch.nn.CrossEntropyLoss,
            batch_size=100,
        )

    # --------------------------------------------------------------------
    def forward(self, X):

        X = self.fc1(X)
        X = self.relu(X)

        X = self.fc2(X)
        X = self.relu(X)

        X = self.fc3(X)
        X = self.relu(X)

        X = self.fc4(X)

        return X

    # --------------------------------------------------------------------
    def reset(self):

        self._score = 0
        self._step = 0
        self._state = []
        return self._env.reset()

    # --------------------------------------------------------------------
    def generateBatches(self):

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_steps = []

        self._stats["actions"] = [0, 0, 0, 0]

        for batch in range(self._batch_size):

            # start a new episode
            state = self.reset()
            states = []
            actions = []
            total_reward = 0
            done = False
            steps = 0

            while not done:

                # run state through the NN and compute most likely action to take
                if self._epsilon < np.random.rand():
                    s = torch.Tensor([state]).float()
                    prob = np.array(self.softmax(self(s))[0].tolist())
                    prob /= prob.sum()
                    action = np.random.choice(len(prob), p=prob)
                else:
                    action = np.random.randint(self._action_size)

                self._stats["actions"][action] += 1

                # perform action and keep track of outcome
                next_state, reward, done, info = self._env.step(action)
                states.append(state)
                actions.append(action)
                total_reward += reward

                # set new state
                state = next_state
                steps += 1

                # add batch for training
                if done:
                    batch_states.append(states)
                    batch_actions.append(actions)
                    batch_rewards.append(total_reward)
                    batch_steps.append(steps)

            if self._epsilon > self._epsilon_min:
                self._epsilon *= self._epsilon_decay

        return batch_states, batch_actions, batch_rewards, batch_steps

    # --------------------------------------------------------------------
    def getBestBatches(self, batch_states, batch_actions, batch_rewards, percentile):

        reward_threshold = np.percentile(batch_rewards, percentile)

        top_states = []
        top_actions = []
        top_rewards = []

        for it in range(len(batch_rewards)):

            if batch_rewards[it] >= reward_threshold:

                top_rewards.append(batch_rewards[it])

                for jt in range(len(batch_states[it])):
                    top_states.append(batch_states[it][jt])
                    top_actions.append(batch_actions[it][jt])

        return top_states, top_actions, top_rewards, reward_threshold

    # --------------------------------------------------------------------
    def run_episode(self, episode=0):

        batch_states, batch_actions, batch_rewards, batch_steps = self.generateBatches()
        top_states, top_actions, top_rewards, threshold = self.getBestBatches(
            batch_states, batch_actions, batch_rewards, percentile=80
        )

        if len(top_states) == 0:
            return -1, -1

        loss, acc, _ = self.train(
            torch.Tensor(np.array(top_states, dtype=np.float32)).float(),
            torch.Tensor(np.array(top_actions, dtype=np.int64)).long(),
        )
        self._stats["loss_batch"].append(loss)
        self._stats["acc_batch"].append(acc)

        self._score = np.mean(top_rewards)
        self._step = np.mean(batch_steps)
        self._threshold = threshold
        if self._score > self._best_score:
            self._best_score = self._score

        self._writer.add_scalar(self._name + "/threshold", self._threshold, episode)
        self._writer.add_scalar(self._name + "/epsilon", self._epsilon, episode)
        self._writer.add_scalar(self._name + "/batch_size", len(top_states), episode)

        mean_loss = np.mean(self._stats["loss_batch"])
        mean_acc = np.mean(self._stats["acc_batch"])

        self._stats["loss_batch"] = []
        self._stats["acc_batch"] = []

        return mean_loss, mean_acc

    # --------------------------------------------------------------------
    def simulate(self):

        state = self._env.reset()
        done = False
        while not done:

            s = torch.Tensor([state]).float()

            probabilities = self(s)
            prob = np.array(probabilities[0].tolist())
            action = np.argmax(prob)

            next_state, reward, done, info = self._env.step(action)

            state = next_state
            self._env.render()


##########################################################################
class LunarLanderTest(unittest.TestCase):

    # --------------------------------------------------------------------
    def testModel(self):

        print("=======================================\n  testModel")

        qt = LunarLander()
        qt.episode(2000, save_best="score", load_best="")

        # qt.loadModel("./output/LunarLander/1552254905.model")
        for runs in range(10):
            qt.simulate()


################################################################################
if __name__ == '__main__':

    unittest.main()

