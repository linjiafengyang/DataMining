import math
import numpy as np
from sklearn.cross_validation import train_test_split

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    x_T = np.transpose(x)
    y_T = np.transpose(y)
    for i in range(0, numIterations):
        hypothesis = sigmoid(np.dot(x, theta))
        loss = hypothesis - y
        # avg cost function J
        cost = 0 - (np.sum(np.dot(y_T, np.log(hypothesis)) + 
                np.dot(1 - y_T, 1 - np.log(hypothesis)))) / m
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(x_T, loss) / m
        # update theta
        theta = theta - alpha * gradient
        print("Iteration %d | Theta: %s" % (i, theta))
    return theta

X = np.array([[1, 1, 0, 1], [0, 1, 0, 0],
              [1, 0, 1, 1], [0, 0, 0, 1],
              [1, 1, 0, 1], [1, 0, 1, 1],
              [1, 0, 0, 0], [1, 0, 1, 1],
              [1, 0, 1, 1], [0, 0, 0, 0],
              [1, 1, 0, 1], [0, 0, 0, 0],
              [1, 0, 0, 1], [0, 0, 0, 0],
              [1, 1, 1, 1], [0, 0, 1, 1],
              [1, 0, 0, 1], [1, 0, 0, 1],
              [0, 0, 0, 1], [0, 0, 0, 1],
              [1, 0, 1, 1], [1, 0, 1, 1],
              [0, 0, 0, 1], [0, 0, 1, 1],
              [1, 0, 1, 1], [0, 0, 0, 0],
              [1, 0, 0, 1], [0, 0, 0, 0],
              [1, 0, 1, 1], [0, 0, 0, 0],
              [1, 0, 0, 1], [1, 0, 1, 1],
              [1, 0, 0, 1], [0, 0, 0, 0],
              [0, 1, 0, 1], [0, 0, 1, 0],
              [1, 1, 0, 1], [1, 1, 0, 0],
              [1, 0, 0, 0], [1, 0, 1, 1]])

m = np.alen(X)
ones = np.ones(m)
X = np.column_stack((ones, X))
Y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
              1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
n = np.alen(X[0])

# 划分为训练集和测试集，测试集占1/5
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)

theta = np.zeros(n)
alpha = 1
numIterations = 43000
theta = gradientDescent(X, Y, theta, alpha, m, numIterations)
print("Theta: ", theta)

predict = sigmoid(np.dot(X_test, theta))
# 将小于0.5的数置为0，将大于等于0.5的数置为1
predict = np.where(predict >= 0.5, predict, 0)
predict = np.where(predict < 0.5, predict, 1)

# 统计预测准确的数目
right = np.sum(predict == Y_test)
# 将预测值和真实值放在一块，便于观察
predict = np.hstack((predict.reshape(-1, 1), Y_test.reshape(-1, 1)))
print("Predict and Y_test: \n", predict)
# 计算在测试集上的准确率
print('测试集准确率：%f%%' % (right * 100.0 / predict.shape[0]))