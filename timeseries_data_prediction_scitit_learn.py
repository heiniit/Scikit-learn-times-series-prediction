# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline

data = pd.read_csv(".\TsData.csv")
data.columns=["date", "value"]
data["year"] = data.apply(lambda row: row["date"][0:4], axis=1)
data["month"] = data.apply(lambda row: row["date"][5:7], axis=1)
data["day"] = data.apply(lambda row: row["date"][8:10], axis=1)
data["hour"] = data.apply(lambda row: row["date"][11:13], axis=1)
data["weekday"] = data.apply(lambda row: date(int(row["year"]),int(row["month"]),int(row["day"])).weekday(), axis=1)

# Take one week apart for testing (middle of the year)
cutindex = 180*24
train1 = data.take(np.arange(0,cutindex))
test = data.take(np.arange(cutindex,cutindex+7*24))
train2 = data.take(np.arange(cutindex+7*24,data.shape[0]))
train = train1.append(train2)

# Modelling
mapper = DataFrameMapper([
    ("hour", None),
    ("day", None),
    ("month", None),
    ("year", None),
    ("weekday", None)
])

# Different kind of models...
pipeline = Pipeline([("mapper", mapper), ("model", NuSVR(kernel="rbf"))])
#pipeline = Pipeline([("mapper", mapper), ("model", MLPRegressor(hidden_layer_sizes=(50,), activation="logistic"))])
#pipeline = Pipeline([("mapper", mapper), ("model", DecisionTreeRegressor(max_depth=5))])

X = train[["hour", "day", "month", "year", "weekday"]]
y = train["value"]

model = pipeline.fit(X = X, y = y)


#Test
test_points = test[["hour", "day", "month", "year", "weekday"]]
test_result = pd.DataFrame(model.predict(test_points))
orig = pd.DataFrame(test["value"]).reset_index(drop=True)

#Compare predicted values with original values that were part of the loaded set
Compare = test_result.join(orig)
Compare.columns=["orig", "predict"]
Compare["diff"] = Compare.apply(lambda row: row["orig"]-row["predict"], axis=1)
#print(Compare)

#Plot
plt.figure(figsize=(20,8))
plt.grid(True)
plt.plot(Compare)
