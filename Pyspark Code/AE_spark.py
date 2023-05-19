from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
import pyspark.sql.functions as F

spark = SparkSession.builder\
.master("local[*]")\
.appName("Spark shell") \
.config("spark.driver.memory", "4g") \
.config("spark.executor.memory", "4g") \
.getOrCreate()

data_points = spark.read.csv("close.csv", header=True, inferSchema=True)
data_points = data_points.drop('Stocks')
data_points = data_points.fillna(-1)

assembler = VectorAssembler(inputCols=data_points.columns, outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])
scaled_data_points = pipeline.fit(data_points).transform(data_points)

(trainingData, validationData) = scaled_data_points.randomSplit([0.9, 0.1])

input_layer = Input(shape=(len(data_points.columns),))
encoded = Dense(13757)(input_layer)
encoded = LeakyReLU()(encoded)
encoded = Dense(10000)(encoded)
encoded = LeakyReLU()(encoded)
encoded = Dense(8000)(encoded)
encoded = LeakyReLU()(encoded)
encoded = Dense(5000)(encoded)
encoded = LeakyReLU()(encoded)
encoded = Dense(1000)(encoded)
encoded = LeakyReLU()(encoded)
encoded = Dense(500)(encoded)
encoded = LeakyReLU()(encoded)
encoded = Dense(100)(encoded)
encoded = LeakyReLU()(encoded)

latent_space = Dense(100, activation="tanh")(encoded)

decoded = Dense(100)(latent_space)
decoded = LeakyReLU()(decoded)
decoded = Dense(500)(decoded)
decoded = LeakyReLU()(decoded)
decoded = Dense(1000)(decoded)
decoded = LeakyReLU()(decoded)
decoded = Dense(5000)(decoded)
decoded = LeakyReLU()(decoded)
decoded = Dense(8000)(decoded)
decoded = LeakyReLU()(decoded)
decoded = Dense(10000)(decoded)
decoded = LeakyReLU()(decoded)
decoded = Dense(13757)(decoded)
decoded = LeakyReLU()(decoded)
decoded = Dense(units=len(data_points.columns), activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", metrics=["mse"], loss="mse")
autoencoder.fit(scaled_data_points.select("scaled_features").withColumnRenamed("scaled_features", "features"),
                scaled_data_points.select("scaled_features").withColumnRenamed("scaled_features", "target"),
                epochs=5000, batch_size=3, validation_split=0.1)

encoder = Model(input_layer, latent_space)
encoded_data_points = encoder.predict(scaled_data_points.select("scaled_features").rdd.map(lambda x: x[0]).collect())
encoded_data_points = [Vectors.dense(row) for row in encoded_data_points]
encoded_data_points = spark.createDataFrame(encoded_data_points, ["features"])

encoded_data_points = data_points.select("id").withColumn("encoded_features", F.explode(encoded_data_points.features))
encoded_data_points = encoded_data_points.select(col("id"), *[col("encoded_features")[i].alias("encoded_"+str(i+1)) for i in range(5)])
encoded_data_points.toPandas().to_csv("output.csv", index=False)
spark.stop()
