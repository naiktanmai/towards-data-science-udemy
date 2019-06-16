#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:28:41 2019

@author: tanmai
"""


import pandas as pd
import numpy as np

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print(X.shape)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

#one hot encoder can be applied only on label encoded vars(i.e numbers)
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

print(y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#backward elimiation
import statsmodels.formula.api as sm
#y = x0b0 + x1+b1 +...
#appending 1 value as b0
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
