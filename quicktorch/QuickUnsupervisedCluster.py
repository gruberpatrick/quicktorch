
import numpy as np

from quicktorch.algorithms.KMeans import KMeans
from quicktorch.algorithms.HierarchicalClustering import HierarchicalClustering

class QuickUnsupervisedCluster:

    _method = ""
    _options = ["k-means", "hierarchical"]
    _values = {}

    # --------------------------------------------------------------------
    def __init__(self, method):

        if method not in self._options:
            print("[ERROR] Available options are: ", ", ".join(self._options))
            return
        self._method = method

    # --------------------------------------------------------------------
    def run(self, data, params={}):

        if self._method == "":
            print("[ERROR] No method set.")
            return -1
        elif self._method == "k-means":
            alg = KMeans()
            return alg.run(data, params)
        elif self._method == "hierarchical":
            alg = HierarchicalClustering()
            return alg.run(data, params)


