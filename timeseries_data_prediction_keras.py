# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# Different amount of data 
#data = pd.read_csv(".\TsData_HalfYear.csv")
data = pd.read_csv(".\TsData_OneYear.csv")
#data = pd.read_csv(".\TsData_FiveYears.csv")


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

# Model creation
def baseline_model():
    model = Sequential()
    model.add(Dense(20, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Fix the seed so we can reproduse similar results
seed = 99
np.random.seed(seed)

# Create the model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=20, batch_size=10, verbose=1)))
pipeline = Pipeline(estimators)

# Train the model
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
