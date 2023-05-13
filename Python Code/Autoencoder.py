import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

data_points = pd.read_csv('data (1).csv')
# print(data_points)
labels = data_points["diagnosis"]
# print(labels)
le = preprocessing.LabelEncoder()
le.fit(labels)
# print(le.classes_)
labels = le.transform(labels)
data_points["diagnosis"] = labels
# # data_points = data_points.iloc[: , :-1]
# print(data_points.dtypes)
# print(data_points.shape)
# print(data_points.isnull())
data_points = data_points.fillna(-1)
data_points = data_points.values

scaler = MinMaxScaler()
df_features = scaler.fit_transform(data_points)

input = Input(shape=df_features.shape[1:])
enc = Dense(33)(input)
enc = LeakyReLU()(enc)
enc = Dense(25)(enc)
enc = LeakyReLU()(enc)
enc = Dense(15)(enc)
enc = LeakyReLU()(enc)
enc = Dense(10)(enc)
enc = LeakyReLU()(enc)
enc = Dense(5)(enc)
enc = LeakyReLU()(enc)

latent_space = Dense(5, activation="tanh")(enc)

dec = Dense(5)(latent_space)
dec = LeakyReLU()(dec)
dec = Dense(10)(dec)
dec = LeakyReLU()(dec)
dec = Dense(15)(dec)
dec = LeakyReLU()(dec)
dec = Dense(25)(dec)
dec = LeakyReLU()(dec)
dec = Dense(33)(dec)
dec = LeakyReLU()(dec)

dec = Dense(units=df_features.shape[1], activation="sigmoid")(dec)
autoencoder = Model(input, dec)
autoencoder.compile(optimizer = "adam",metrics=["mse"],loss="mse")
autoencoder.fit(df_features, df_features, epochs=5000, batch_size=32, validation_split=0.1)

encoder = Model(input, latent_space)
test_au_features = encoder.predict(df_features)
print(test_au_features.shape)
test_au_features = pd.DataFrame(test_au_features)
test_au_features.to_csv("output.csv")