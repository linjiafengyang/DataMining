import numpy as np
from numpy import *
from time import sleep  
from matplotlib import pyplot as plt
  
#加载文本文件中的数据，返回数据矩阵和类标签  
def loadDataSet(fileName):  
    dataMat = []; labelMat = []  
    fr = open(fileName)  
    for line in fr.readlines():  
        lineArr = line.strip().split('\t')  
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  
        labelMat.append(float(lineArr[2]))  
    return dataMat,labelMat  
  
#随机选择不等于i的值。i是第一个α的下表，m是α的数量  
def selectJrand(i,m):  
    j=i #we want to select any J not equal to i  
    while (j==i):  
        j = int(random.uniform(0,m))  
    return j  
  
#调整大于H或小于L的α值  
def clipAlpha(aj,H,L):  
    if aj > H:  
        aj = H  
    if L > aj:  
        aj = L  
    return aj

#dataMatIn, classLabels, C, toler, maxIter分别为数据集、类标签、常数C、容错率和最大迭代次数  
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()  
    b = 0; m,n = shape(dataMatrix)  
    alphas = mat(zeros((m,1)))  
    iter = 0  
    while (iter < maxIter):  
        alphaPairsChanged = 0  
        for i in range(m):  
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  
            Ei = fXi - float(labelMat[i])  
               #if checks if an example violates KKT conditions  
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):  
                j = selectJrand(i,m)  
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b  
                Ej = fXj - float(labelMat[j])  
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();  
                #将α的值调整到区间[0,C]  
                if (labelMat[i] != labelMat[j]):  
                    L = max(0, alphas[j] - alphas[i])  
                    H = min(C, C + alphas[j] - alphas[i])  
                else:  
                    L = max(0, alphas[j] + alphas[i] - C)  
                    H = min(C, alphas[j] + alphas[i])  
                if L==H: print ("L==H"); continue  
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T  
                if eta >= 0: print ("eta>=0"); continue  
                #对αi和αj进行修改，αi的修改量和αj相同，但方向相反  
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta  
                alphas[j] = clipAlpha(alphas[j],H,L)  
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue  
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j  
                                                                        #the update is in the oppostie direction  
                #b的更新  
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T  
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T  
                if (0 < alphas[i]) and (C > alphas[i]): b = b1  
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2  
                else: b = (b1 + b2)/2.0  
                alphaPairsChanged += 1  
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))  
        if (alphaPairsChanged == 0): iter += 1  
        else: iter = 0  
        print ("iteration number: %d" % iter  )
    return b,alphas

dataMat,labelMat = loadDataSet('smo.txt')
C = 1
toler = 0.001
maxIter = 200
b,alphas = smoSimple(dataMat, labelMat, C, toler, maxIter)
print(alphas)
print(b)

alphas = np.array(alphas)
labelMat = np.array(labelMat)

theta12 = (alphas.T*labelMat).dot(dataMat)
print(theta12)
theta0 = theta12[0,0]
theta1 = theta12[0,1]
k = -theta0 / theta1
b = -b / theta1
print("k: %.1f" % k)
print("b: %.1f" % b)
print("最优超平面（决策边界）的方程：y = %.1fx + %.1f" % (k, b))
