import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def loadTrainData(path):
    train_data = h5py.File(path)
    train_feat = train_data['train_feat']
    train_feat = np.transpose(train_feat)
    train_label = train_data['train_label']
    train_label = np.transpose(train_label)
    train_label = np.ravel(train_label)
    print(train_feat.shape)
    print(train_label.shape)
    return train_feat, train_label

def loadTestData(path):
    test_data = h5py.File(path)
    test_feat = test_data['test_feat']
    test_feat = np.transpose(test_feat)
    print(test_feat.shape)
    return test_feat

def saveResult(test_label):
    res = pd.read_csv('data/submission_sample.csv')
    image = res.iloc[:, [0]]
    data = np.hstack((image, test_label))
    df = pd.DataFrame(data, columns=['image', 'label'])
    df.to_csv('15331046_czh.csv', index=False)

train_data_path = 'data/train_data.mat'
train_feat, train_label = loadTrainData(train_data_path)

# x_mean = train_feat.mean(axis=0)
# x_std = train_feat.std(axis=0)
# X = (train_feat - x_mean) / x_std
scaler = StandardScaler()
scaler.fit(train_feat)
X = scaler.transform(train_feat)

print('Start training...')
clf = MLPClassifier(hidden_layer_sizes=(5400,), activation='logistic', 
        solver='sgd', batch_size=200, learning_rate_init=0.1)
clf.fit(X, train_label)
print(clf.score(X, train_label))
print(clf.loss_)

test_data_path = 'data/test_data_raw.mat'
test_feat = loadTestData(test_data_path)
test_feat_scaled = scaler.transform(test_feat)

# print('Load model...')
# clf = joblib.load('mlp_5400.m')

print('Start testing...')
test_label = clf.predict(test_feat_scaled)

test_label = pd.DataFrame(test_label, columns=['label'], dtype=int)
print(test_label.shape)
saveResult(test_label)
