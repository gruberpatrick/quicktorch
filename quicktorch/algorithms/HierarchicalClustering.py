
import numpy as np

class HierarchicalClustering:
    
    _header = []
    _data = []
    _distance = ["single_link", "complete_link", "average_link"]
    _elements = -1
    _similarity_measure = ""
    _k = -1
    _clusters = []
    _cluster_tracker = {}

    # -------------------------------------------------------------------
    def __init__(self):

        self.initializeAgglomarativeClusters()

    # -------------------------------------------------------------------
    def initializeAgglomarativeClusters(self):

        if self._similarity_measure == "single_link": self.buildGraph()
        for elem in range(self._elements):
            self._clusters.append([elem])
            self._cluster_tracker[elem] = elem

    # -------------------------------------------------------------------
    def buildGraph(self):

        self._distances = []

        for it in range(len(self._data)):
            for jt in range(it + 1, len(self._data)):
                self._distances.append([it, jt, self.euclideanDistance(it, jt)])

        self._distances.sort(key=lambda elem: elem[2])

    # -------------------------------------------------------------------
    def euclideanDistance(self, data1_index, data2_index):

        ssum = 0
        for dim in range(len(self._data[data1_index])):
            ssum += (self._data[data1_index][dim] - self._data[data2_index][dim])**2

        return np.sqrt(ssum)

        """return math.sqrt(
            (self._data[data1_index][0] - self._data[data2_index][0])**2 + \
            (self._data[data1_index][1] - self._data[data2_index][1])**2
        )"""

    # -------------------------------------------------------------------
    def returnDistances(self, distances, arg=False):

        if self._similarity_measure == "single_link":
            return min(distances)
        elif self._similarity_measure == "complete_link":
            return max(distances)
        elif self._similarity_measure == "average_link":
            return sum(distances) / len(distances)
            
        return -1

    # -------------------------------------------------------------------
    def clusterDistance(self, cluster1_index, cluster2_index):

        c1 = self.getCluster(cluster1_index)
        c2 = self.getCluster(cluster2_index)

        distances = []

        for elem_c1 in c1:
            for elem_c2 in c2:
                distances.append( self.euclideanDistance(elem_c1, elem_c2) )

        return self.returnDistances(distances)

    # -------------------------------------------------------------------
    def getMinArg(self, distances):
        
        if len(distances) == 0: return -1
        return min(range(len(distances)), key=distances.__getitem__)

    # -------------------------------------------------------------------
    def mergeClusters(self, cluster1_index, cluster2_index):

        c1 = set(self.getCluster(cluster1_index))
        return list(c1.union(set(self.getCluster(cluster2_index))))

    # -------------------------------------------------------------------
    def getCluster(self, cluster_index):

        try:
            return self._clusters[cluster_index][:]
        except:
            return []

    # -------------------------------------------------------------------
    def compareClusterDistance(self):

        clusters = []
        merges = {}
        dist_min = float("inf")
        merger = []

        for it in range(len(self._clusters)):
            
            distance = []
            for jt in range(len(self._clusters)):
                if it == jt: distance.append(float("inf"))
                else: distance.append(self.clusterDistance(it, jt))
            
            merge = self.getMinArg(distance)
            dist = min(distance)
            if dist < dist_min:
                dist_min = dist
                merger = [it, merge]

        clusters.append(self.mergeClusters(merger[0], merger[1]))
        merges[merger[0]] = 1
        merges[merger[1]] = 1

        for it in range(len(self._clusters)):
            if it not in merges: clusters.append(self.getCluster(it))

        self._clusters = clusters

    # -------------------------------------------------------------------
    def increasedPerformanceCluster(self):

        clusters = []
        merges = {}
        val = self._distances.pop(0)

        merger = val[0:2]

        if self._cluster_tracker[merger[0]] == self._cluster_tracker[merger[1]]: return

        merged_cluster = self.mergeClusters(self._cluster_tracker[merger[0]], self._cluster_tracker[merger[1]])
        clusters.append(merged_cluster)
        merges[self._cluster_tracker[merger[0]]] = 1
        merges[self._cluster_tracker[merger[1]]] = 1

        for elem in merged_cluster: self._cluster_tracker[elem] = 0

        for it in range(len(self._clusters)):
            cluster = self.getCluster(it)
            if it not in merges:
                clusters.append(cluster)
                for elem in cluster: self._cluster_tracker[elem] = len(clusters) - 1

        self._clusters = clusters[:]

    # -------------------------------------------------------------------
    def exportClusters(self):

        enc = {}
        c = 0
        classes = []

        for cluster in self._clusters:
            for elem in cluster:
                enc[elem] = c
            c += 1

        for it in range(self._elements):
            classes.append(enc[it])

        return classes

    # -------------------------------------------------------------------
    def run(self, data, params={}):

        if "method" not in params or "k" not in params:
            print("[ERROR] K-Means requires parameter 'k' and 'method'.")
            return -1

        if params["method"] not in self._distance:
            print("[ERROR] Valid options for 'method' are:", ",".join(self._distance))
            return -1

        self._similarity_measure = params["method"]
        self._k = params["k"]
        self._data = data
        self._elements = self._data.shape[0]

        it = 0
        while len(self._clusters) > self._k:
            print("\r\tRound", str(it), end="")
            if self._similarity_measure != "single_link": self.compareClusterDistance()
            else: self.increasedPerformanceCluster()
            print("  - Clusters:", len(self._clusters))
            it += 1

        return self.exportClusters()
        

