import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

class Particle:
    def __init__(self, n_clusters, data, w=0.1, c1=1.5, c2=1.5):
        self.n_clusters = n_clusters
        self.centroids_pos = np.array(data.takeSample(False, self.n_clusters))

        self.pb_val = np.inf
        self.pb_pos = self.centroids_pos.copy()
        self.velocity = np.zeros_like(self.centroids_pos)
        self.pb_clustering = None
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_pb(self, data):
        distances = self._get_distances(data=data)
        clusters = np.argmin(distances, axis=0)

        new_val = self._fitness_function(clusters=clusters, distances=distances)
        if new_val < self.pb_val:
            self.pb_val = new_val
            self.pb_pos = self.centroids_pos.copy()
            self.pb_clustering = clusters.copy()

    def update_velocity(self, gb_pos):
        self.velocity = self.w * self.velocity + self.c1 * np.random.random() * (self.pb_pos - self.centroids_pos) + self.c2 * np.random.random() * (gb_pos - self.centroids_pos)

    def move_centroids(self, gb_pos):
        self.update_velocity(gb_pos=gb_pos)
        new_pos = self.centroids_pos + self.velocity
        self.centroids_pos = new_pos.copy()

    def _get_distances(self, data):
        distances = data.rdd.flatMap(lambda x: [(Vectors.dense(x), Vectors.dense(cp)) for cp in self.centroids_pos]).map(lambda x: (x[0], x[1], float(x[0].squared_distance(x[1])))).toDF(["data", "centroid", "distance"])
        min_distance_udf = udf(lambda a, b: float(np.min(a)) if np.min(a) > 0 else b, DoubleType())
        distances = distances.groupBy("data").agg(min_distance_udf("distance", np.inf).alias("min_distance"))
        distances = np.array(distances.select("min_distance").rdd.flatMap(lambda x: x).collect())
        distances = np.reshape(distances, (self.n_clusters, len(data)))
        return distances

    def _fitness_function(self, clusters, distances):
        J = 0.0
        for i in range(self.n_clusters):
            p = np.where(clusters == i)[0]
            if len(p):
                d = sum(distances[i][p])
                d /= len(p)
                J += d
        J /= self.n_clusters
        return J
