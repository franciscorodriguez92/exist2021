# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#from sklearn.svm import LinearSVC
from sklearn.svm import SVC



def get_classifier(method='logistic_regression'):
    if (method in 'logistic_regression') or (method in 'baseline'):
        return LogisticRegression(C=1e3,
                                  tol=0.01,
                                  multi_class='ovr',
                                  solver='liblinear',
                                  n_jobs=-1,
                                  random_state=123)
    if 'random_forest' == method:
        return RandomForestClassifier(n_estimators=250,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=123)

    if 'svm' == method:
        #return SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
        return SVC(kernel="linear",probability=True,class_weight=None, random_state=123)


