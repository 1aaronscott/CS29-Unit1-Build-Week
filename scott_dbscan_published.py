''' Class implementation of the DBSCAN algorithm '''
import numpy as np
from scipy.spatial.distance import squareform, pdist


"""[From Wikipedia](https://en.wikipedia.org/w/index.php?title=DBSCAN&oldid=958031760) the algorithm can be expressed in pseudocode as follows:


```
DBSCAN(DB, distFunc, eps, minPts) {
    C = 0                                                  /* Cluster counter */
    for each point P in database DB {
        if label(P) ≠ undefined then continue              /* Previously processed in inner loop */
        Neighbors N = RangeQuery(DB, distFunc, P, eps)     /* Find neighbors */
        if |N| < minPts then {                             /* Density check */
            label(P) = Noise                               /* Label as Noise */
            continue
        }
        C = C + 1                                          /* next cluster label */
        label(P) = C                                       /* Label initial point */
        Seed set S = N \ {P}                               /* Neighbors to expand */
        for each point Q in S {                            /* Process every seed point */
            if label(Q) = Noise then label(Q) = C          /* Change Noise to border point */
            if label(Q) ≠ undefined then continue          /* Previously processed */
            label(Q) = C                                   /* Label neighbor */
            Neighbors N = RangeQuery(DB, distFunc, Q, eps) /* Find neighbors */
            if |N| ≥ minPts then {                         /* Density check */
                S = S ∪ N                                  /* Add new neighbors to seed set */
            }
        }
    }
}
```
where RangeQuery can be implemented using a database index for better performance, or using a slow linear scan:
```
RangeQuery(DB, distFunc, Q, eps) {
    Neighbors = empty list
    for each point P in database DB {                      /* Scan all points in the database */
        if distFunc(Q, P) ≤ eps then {                     /* Compute distance and check epsilon */
            Neighbors = Neighbors ∪ {P}                    /* Add to result */
        }
    }
    return Neighbors
}
```
"""


class scott_dbscan():
    ''' Homegrown implementation of the DBSCAN algorithm '''

    def __init__(self, points, distance=.1, minpoints=5, distance_metric='euclidean'):
        ''' Do the clustering of points
        Input:  points: a (vector scaled?) array of points to be clustered
                distance: the distance between two points such that 
                they are considered connected
                minpoints: the minimum number of points to define a 
                cluster
                distance_metric: how to measure the distance between points;
                see scipy.spatial.distance.pdist docs '''

        self.points = points
        self.distance = distance
        self.minpoints = minpoints
        self.distance_metric = distance_metric
        self.m = self.points.shape[0]  # number of datapoints in points
        # don't eval points already processed
        self.seen_already = np.zeros(self.m, 'intc')
        self.point_type = np.zeros(self.m)  # 1=core, 0=border, -1=noise
        self.cluster_points = []  # points in a cluster
        self.labels = np.zeros(self.m)  # cluster assignment
        self.cluster_counter = 1  # number of cluster
        self.closest_points = []  # points closest to current point

    def fit(self):
        ''' Perform the clustering of data
        Output: self object '''
        # calc pairwise distancees between points and put into an array
        self.square_matrix = squareform(
            pdist(self.points, self.distance_metric))
        for i in range(self.m):
            # look only at points not already looked at for clustering purposes
            if self.seen_already[i] == 0:
                self.seen_already[i] = 1
                # find close points
                self.closest_points = np.where(
                    self.square_matrix[i] < self.distance)[0]
                # if there aren't enough point close by it's a noise point
                if len(self.closest_points) < self.minpoints:
                    self.point_type[i] = -1
                    continue
                else:
                    self.cluster_points.append(i)
                    self.labels[i] = self.cluster_counter
                    self.closest_points = list(set(self.closest_points))
                    self.grow_cluster()
                    self.cluster_points.append(self.closest_points[:])
                    self.cluster_counter += 1

        return self

    def grow_cluster(self):
        ''' Enlarge the cluster around a given point '''
        self.Neighbors = []

        # scan all points in the database
        for i in self.closest_points:
            # if the point hasn't been seen already
            if self.seen_already[i] == 0:
                self.seen_already[i] = 1
                self.Neighbors = np.where(
                    self.square_matrix[i] < self.distance)[0]
                # if point is close enough add to closest_points
                if len(self.Neighbors) >= self.minpoints:
                    for j in self.Neighbors:
                        try:
                            self.closest_points.index(j)
                        except ValueError:
                            self.closest_points.append(j)

            # add points to clusters and cluster labels
            if self.labels[i] == 0:
                self.cluster_points.append(i)
                self.labels[i] = self.cluster_counter
        return self.Neighbors

    def fit_predict(self, points, distance=.1, minpoints=5, distance_metric='euclidean'):
        ''' Method that returns the labels (clusters) of the fit method '''
        self.fit()
        return self.labels
