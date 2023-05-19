import numpy as np
from pyspark.sql import SparkSession
from pso_spark import PSOClustering
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder\
.master("local[*]")\
.appName("Spark shell") \
.config("spark.driver.memory", "4g") \
.config("spark.executor.memory", "4g") \
.getOrCreate()

data_points = spark.read.csv('output1.csv', header=True, inferSchema=True)
assembler = VectorAssembler(inputCols=data_points.columns, outputCol="features")

pso = PSOClustering(n_clusters=10, n_particles=8, data=data_points)
pso.start(iteration=1000)
spark.stop()