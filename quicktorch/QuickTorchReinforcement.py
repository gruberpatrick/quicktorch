import os
import numpy as np
from QuickTorch import QuickTorch
from tensorboardX import SummaryWriter


class QTReinforcement(QuickTorch):

    # -----------------------------------------------------------
    def run_episode(self, episode=None):

        print("[ERROR] Function 'run_episode' not implemented.")
        return -1, -1

    # -----------------------------------------------------------
    def episode(self, episodes, save_best="", load_best=""):

        self.createProjectFolder()

        best = {"score": -np.inf, "step": -np.inf, "trigger": []}
        scores = []
        try:

            for episode in range(episodes):

                best["trigger"] = []

                # run reinforcement learning;
                loss, _ = self.run_episode(episode=episode)
                scores.append(self._score)

                # save the current best scores;
                if self._score > best["score"]:
                    best["score"] = self._score
                    best["trigger"].append("score")
                if self._step > best["step"]:
                    best["step"] = self._step
                    best["trigger"].append("step")

                # output statements;
                print(
                    "[%5d / %5d] Score: %8.4f - Steps: %8.4f - Best Score: %8.4f"
                    % ((episode + 1), episodes, self._score, self._step, self._best_score)
                )

                # save best results;
                if save_best != "" and save_best in best["trigger"]:
                    self.saveModel()
                    print("  New model saved...")

                if load_best == "score" and self._score < best["score"]:
                    try:
                        self.loadModel("./output/" + self._name + "/" + self._timestamp + ".model")
                        print("  Model reset...")
                        continue
                    except Exception:
                        pass

                self._writer.add_scalar(self._name + "/score", self._score, episode)
                self._writer.add_scalar(self._name + "/steps", self._step, episode)
                self._writer.add_scalar(self._name + "/loss", loss, episode)

        except KeyboardInterrupt:
            print(" Finishing...")
