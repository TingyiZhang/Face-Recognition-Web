import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics

data = np.load('./data_pca_50.pickle.npz')

X = data[data.files[0]]
y = data[data.files[1]]
mean = data[data.files[2]]

# Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# SVM
# model = SVC(kernel='rbf', gamma=0.01, probability=True)
# model.fit(x_train, y_train)

# score
# print(model.score(x_test, y_test))
# y_pred = model.predict(x_test)
# y_prob = model.predict_proba(x_test)  # probability
# cm = metrics.confusion_matrix(y_test, y_pred)
# print(cm)

model_tune = SVC()
param_grid = {'C': [1, 10, 20, 30, 50, 100],
              'kernel': ['rbf', 'poly'],
              'gamma': [0.1, 0.05, 0.001, 0.002, 0.005], 'coef0': [0, 1]}
model_grid = GridSearchCV(model_tune, param_grid=param_grid, scoring='accuracy', cv=5, verbose=True)
model_grid.fit(X, y)

best_params = model_grid.best_params_
model_best = SVC(C=best_params['C'],
                 kernel=best_params['kernel'],
                 gamma=best_params['gamma'],
                 coef0=best_params['coef0'], probability=True)

model_best.fit(x_train, y_train)
pickle.dump(model_best, open('./svm_model.pickle', 'wb'))
pickle.dump(mean, open('./mean.pickle', 'wb'))