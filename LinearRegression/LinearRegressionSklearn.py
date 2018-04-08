import numpy as np
from sklearn import linear_model

"""
scikit-learn
"""

Y = np.array([89, 91, 93, 95, 97])
X = np.array([[87, 72, 83, 90],
              [89, 76, 88, 93],
              [89, 74, 82, 91],
              [92, 71, 91, 89],
              [93, 76, 89, 94]])
m = np.alen(X)
ones = np.ones(m)
X = np.column_stack((ones, X))

model = linear_model.LinearRegression()
model.fit(X, Y)
x_predict = np.array([[1, 88, 73, 87, 92]])
result = model.predict(x_predict)
print("Predit result: ", result)