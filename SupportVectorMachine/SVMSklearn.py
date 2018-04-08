import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm 

    
# 作图
def plot_data(X, y):
    plt.figure(figsize=(10, 8))
    pos = np.where(y == 1) # 找到y==1的位置
    neg = np.where(y == 0) # 找到y==0的位置
    # np.ravel使多维矩阵降为一维
    p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(X[pos, 1]), 'rx', markersize=8)
    p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(X[neg, 1]), 'bo', markersize=8)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend([p1, p2], ["y==+", "y==—"])
    return plt
    

def SVM(X, y, kernels='linear'):
    '''线性分类'''
    model = svm.SVC(C=1.0, kernel='linear').fit(X, y) # 指定核函数为线性核函数
    '''非线性分类'''
    # model = svm.SVC(gamma=100).fit(X, y) # gamma为核函数的系数，值越大拟合的越好

    plt = plot_data(X, y)
    
    # 画线性分类的决策边界
    if kernels == 'linear':
        theta12 = model.coef_ # theta1 and theta2
        theta0 = model.intercept_ # theta0
        print("Theta 1 and theta 2: ", theta12)
        print("Theta 0: ", theta0)
        print("k: ", -(theta12[0, 0] * 100) / (100 * theta12[0, 1]))
        print("b: ", float(-theta0 * 100 / (theta12[0, 1] * 100)))
        print("最优超平面（决策边界）的方程：y = %dx + %.1f" % (-(theta12[0, 0] * 100) / (100 * theta12[0, 1]), -theta0 / theta12[0, 1]))
        x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2 = -(theta12[0, 0] * x1 + theta0) / theta12[0, 1] # theta0 + theta1*x1 + theta2*x2 == 0
        plt.plot(x1, x2, 'green', linewidth=2.0)
        plt.show()
    # 画非线性分类的决策边界
    else:
        x_1 = np.transpose(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).reshape(1, -1))
        x_2 = np.transpose(np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(1, -1))
        X1, X2 = np.meshgrid(x_1, x_2)
        vals = np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            this_X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
            vals[:, i] = model.predict(this_X)
        plt.contour(X1, X2, vals, [0, 1], color='green')
        plt.show()

X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
y = np.array([1, 1, 1, 0, 0, 0])

plot_data(X, y)
SVM(X, y) # 线性分类画决策边界
# SVM(X, y, model, class_='notLinear') # 非线性分类画决策边界