import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
# Sample data
X = [[1], [2], [3], [4], [5]]  # input
y = [2, 4, 6, 8, 10]           # output

model = LinearRegression()
model.fit(X, y)

print("Prediction for 6:", model.predict([[6]]))
