from scipy.io import loadmat
import numpy as np

def cost(X, Theta, Y, R, lamda, learning_rate, num_iterations):
    for i in range(0, num_iterations):
        # compute the cost
        error = np.multiply(np.dot(X, Theta.T) - Y, R)
        squared_error = np.power(error, 2)
        print("迭代次数 %d | 均方误差: %f" % (i+1, np.sum(squared_error)))
        J = (1 / 2) * np.sum(squared_error) + \
            ((lamda / 2) * np.sum(np.power(Theta, 2))) + \
            ((lamda / 2) * np.sum(np.power(X, 2)))
        # calculate the gradients with regularization
        X = X - learning_rate * ((error * Theta) + (lamda * X))
        Theta = Theta - learning_rate * ((error.T * X) + (lamda * Theta))
    return X, Theta

Y = np.matrix([[4, 4, 0, 0, 1, 1, 5, 0],
     [5, 5, 0, 1, 0, 0, 0, 0],
     [0, 4, 1, 0, 0, 1, 5, 4],
     [5, 0, 2, 5, 0, 1, 2, 0],
     [1, 0, 5, 4, 5, 0, 0, 1],
     [1, 0, 5, 0, 0, 4, 0, 0],
     [0, 1, 0, 5, 0, 5, 1, 0]])
R = np.where(Y == 0, Y, 1)

num_movies = Y.shape[0]
num_users = Y.shape[1]
num_features = 4

X = np.random.random((num_movies, num_features))
Theta = np.random.random((num_users, num_features))
X = np.ones((num_movies, num_features))
Theta = np.ones((num_users, num_features))

lamda = 0.1
learning_rate = 0.01
num_iterations = 23000

X, Theta = cost(X, Theta, Y, R, lamda, learning_rate, num_iterations)
print("X为:\n", np.around(X, decimals=4))
print("Theta为:\n", np.around(Theta, decimals=4))
result = np.dot(X, Theta.T)
result = np.around(result, decimals=1)
print("效用矩阵为:\n", result)