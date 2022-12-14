# -*- coding: utf-8 -*-
"""ML_project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11vhff5DCPDlGy3alSVm8gniC3vrYH6XE
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_processing import process_data

X_train, X_test, y_train, y_test, airlines_mapping, airplanes_mapping = process_data("flight_data.csv")

model = LinearRegression(random_state = 0).fit(X_train, y_train)

y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)

for col in X:
  plt.subplot(2, 3, i)

  X_feature = []

  for row in col:
    X_feature.append(row)

  plt.scatter(X_feature, y_pred)
  # plt.plot(X, model.predict(X))

plt.show()