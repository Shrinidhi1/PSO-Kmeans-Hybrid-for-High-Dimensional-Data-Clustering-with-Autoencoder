import pandas as pd
from pso_clustering import PSOClustering
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

data_points = pd.read_csv('output1.csv')
# print(data_points)
# labels = data_points["diagnosis"]
# print(labels)
# le = preprocessing.LabelEncoder()
# le.fit(labels)
# print(le.classes_)
# labels = le.transform(labels)
# data_points["diagnosis"] = labels
# print(data_points.dtypes)
# print(data_points.shape)
# print(data_points.isnull())
# data_points = data_points.fillna(-1)
data_points = data_points.values

# data_points = pd.read_csv('output.csv')
# data_points = data_points.values

pso = PSOClustering(n_clusters=10, n_particles=8, data=data_points)
pso.start(iteration=1000)