import numpy as np

"""
正则化
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
X_T = np.transpose(X)
lamda = 1
matrix = np.eye(np.alen(X))
matrix[0][0] = 0
# print(matrix)



# 无正则化
# theta = (X'X)^(-1)X'Y
# temp1 = np.dot(X_T, X)

# L2正则化
# theta = (X'X + lamda*matrix)^(-1)X'Y
temp1 = np.dot(X_T, X) + lamda * matrix

temp2 = np.linalg.inv(temp1)
temp3 = np.dot(temp2, X_T)
theta = np.dot(temp3, Y)
print("Theta: ", theta)

x_predit = [1, 88, 73, 87, 92]
print("Predit result: ", np.dot(x_predit, theta))
