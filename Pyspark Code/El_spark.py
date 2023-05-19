from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder\
.master("local[*]")\
.appName("Spark shell") \
.config("spark.driver.memory", "4g") \
.config("spark.executor.memory", "4g") \
.getOrCreate()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(2, 10)

data_points = spark.read.csv('output.csv', header=True, inferSchema=True)
# data_points = data_points.drop("ID", "molecule_name", "conformation_name")
assembler = VectorAssembler(inputCols=data_points.columns, outputCol="features")
data = assembler.transform(data_points).select("features")

for k in K:
    kmeans = KMeans().setK(k).setSeed(1)
    model = kmeans.fit(data)

    # Make predictions
    predictions = model.transform(data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    distortions.append(silhouette)

    # Within Set Sum of Squared Errors
    wssse = model.computeCost(data)
    inertias.append(wssse)

    mapping1[k] = silhouette
    mapping2[k] = wssse

for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

for key, val in mapping2.items():
    print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

spark.stop()