#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from keras.layers import Dense
from keras.models import Sequential


model = Sequential()
model.add(Dense(units=20, activation="sigmoid", input_dim=4))
model.add(Dense(units=3, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


model.summary()


df = pd.read_csv("iris.csv")
df.head()


X = df.drop("class", axis=1)
y = df["class"]


onehot_enc = OneHotEncoder(sparse=False)
y = onehot_enc.fit_transform(y.values.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


history = model.fit(X_train, y_train, epochs=500)


predictions = model.predict(X_test)


predictions = onehot_enc.inverse_transform(predictions)


y_test = onehot_enc.inverse_transform(y_test)


f1_score(y_test, predictions, average="weighted")


accuracy_score(y_test, predictions)