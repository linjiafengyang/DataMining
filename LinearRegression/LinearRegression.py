import numpy as np

"""
梯度下降法
"""

# features scaling
def featuresNormalization(x):
    x_mean = np.mean(x, axis=0) # 列均值
    x_max = np.max(x, axis=0) # 列最大值
    x_min = np.min(x, axis=0) # 列最小值
    x_s = x_max - x_min
    x = (x - x_mean) / x_s
    return x, x_mean, x_s

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    x_T = np.transpose(x)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost function J
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(x_T, loss) / m
        # update theta
        theta = theta - alpha * gradient
        print("Iteration %d | Theta: %s" % (i, theta))
    return theta


Y = np.array([89, 91, 93, 95, 97])
X = np.array([[87, 72, 83, 90],
              [89, 76, 88, 93],
              [89, 74, 82, 91],
              [92, 71, 91, 89],
              [93, 76, 89, 94]])
X, X_mean, X_s = featuresNormalization(X) # 特征值缩放
Y = np.array([89, 91, 93, 95, 97])
m = np.alen(X)
ones = np.ones(m)
X = np.column_stack((ones, X))
n = np.alen(X[0])
alpha = 1
theta = np.zeros(n)

print(theta)
print(X)

# 6354可达到与sklearn相同的预测值
theta = gradientDescent(X, Y, theta, alpha, m, 5)
print("Theta: ", theta)

x_predict = np.array([[88, 73, 87, 92]])
x_predict = (x_predict - X_mean) / X_s
m = np.alen(x_predict)
ones = np.ones(m)
x_predict = np.column_stack((ones, x_predict))
result = np.dot(x_predict, theta)
print("Predit result: %.4f" % result)
