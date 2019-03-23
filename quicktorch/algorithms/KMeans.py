
import numpy as np

class KMeans:

    _method = ""
    _options = ["k-means"]
    _values = {}

    # --------------------------------------------------------------------
    def initializeCentroids(self, data, k, random=True):

        centroids = []

        if random: # random initialization;
            for jt in range(data.shape[1]):
                init = np.random.uniform(np.min(data[:,jt]), np.max(data[:,jt]), k).tolist()
                centroids.append(init)
                centroids = np.array(centroids).T
        else: # k-means++;
            centroid = []
            for jt in range(data.shape[1]): centroid.append(np.random.uniform(np.min(data[:,jt]), np.max(data[:,jt]), 1)[0])
            centroids.append(centroid)
            while len(centroids) <= k - 1:
                _, _, class_distances = self.assignCluster(data, np.array(centroids))
                maxx = np.argmax(class_distances, axis=1)
                centroids.append(data[maxx[-1]].tolist())
            centroids = np.array(centroids)

        return centroids

    # --------------------------------------------------------------------
    def assignCluster(self, data, centroids):

        # calculate distances;
        dist = []   
        for centroid in centroids.tolist():
            dist.append( (data - centroid)**2 )
        dist = np.array(dist)

        # addign class;
        class_distances = np.sqrt(dist.sum(axis=2))
        return np.argmin(class_distances, axis=0), dist, class_distances
        
    # --------------------------------------------------------------------
    def recalulateCentroids(self, data, centroids, classes):

        for it in range(centroids.shape[0]):

            cluster = data[classes == it]
            if cluster.shape[0] == 0: continue
            mean = np.mean(cluster, axis=0)
            centroids[it] = mean.tolist()

        return np.array(centroids)

    # --------------------------------------------------------------------
    def kMeans(self, data, params):

        # check params;
        if "k" not in params:
            print("[ERROR] K-Means requires parameter 'k'.")
            return -1

        # initialize centroids;
        centroids = self.initializeCentroids(data, params["k"], random=False)
        self._values["centroids"] = centroids.T
        classes = None
        
        # perform step
        for it in range(1000):
            print("STEP", it+1)
            classes, _, _ = self.assignCluster(data, centroids)
            centroids = self.recalulateCentroids(data, centroids, classes)
            self._values["centroids"] = centroids.T

        return classes

    # --------------------------------------------------------------------
    def run(self, data, params={}):

        return self.kMeans(data, params)


