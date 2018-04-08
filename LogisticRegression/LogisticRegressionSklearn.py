from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import numpy as np

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
# 划分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 逻辑回归
model = LogisticRegression()
model.fit(X_train, Y_train)
print("Theta: ", model.coef_)

# 预测
predict = model.predict(X_test)
right = sum(predict == Y_test)
# 将预测值和真实值放在一块，便于观察
predict = np.hstack((predict.reshape(-1, 1), Y_test.reshape(-1, 1)))
print("Predict and Y_test: \n", predict)
# 计算在测试集上的准确度
print('测试集准确率：%f%%' % (right*100.0 / predict.shape[0]))
