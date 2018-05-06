import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.multiclass import OutputCodeClassifier, OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv", header=None)
# remove the non-numeric columns
train_data = train_data._get_numeric_data()
# create a numpy array with the numeric values for input into scikit-learn
to_nparray = train_data.as_matrix()
# 从第一列开始到倒数第二列
X = to_nparray[:,0:-1]
# 最后一列
y = to_nparray[:,-1]

test_data = pd.read_csv("test_raw.csv", header=None)
# 第一列
ID = test_data.iloc[:, [0]]
print(ID.shape)
# remove the non-numeric columns
test_data = test_data._get_numeric_data()
# create a numpy array with the numeric values for input into scikit-learn
test_data = test_data.as_matrix()

# clf = OutputCodeClassifier(LinearSVC(random_state=0),
#                             code_size=2, random_state=0)
#clf = OneVsOneClassifier(LinearSVC(random_state=0))
#clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)

res = clf.fit(X, y).predict(test_data)
res = res.reshape((res.shape[0], 1))
print(res.shape)
print(clf.score(X, y))

prefix = np.array(['cls_'])
# np.dtype: int转字符串
res = np.char.mod('%d', res)
# numpy拼接字符串
res = np.core.defchararray.add(prefix, res)
# 按行合并两个numpy.array
data = np.hstack((ID, res))
# numpy.array转DataFrame
df = DataFrame(data, columns=['ID', 'Pred'])
# 忽略index
df.to_csv("result.csv", index=False)
