# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:23:03 2015

@author: Vic
"""
import sklearn.linear_model
from sklearn.metrics import log_loss
from sklearn.metrics import auc

from logisticregression import *
from sklearn import datasets


def eval_my_lr(X, y):
    my_lr = LogisticRegression(reg_param=0.5)
    my_lr.fit(X, y)
    pred = my_lr.predict(X)

    print "los loss={}".format(log_loss(y, pred))
    print 'my coef={}'.format(my_lr.coef)


def eval_sklearn_lr(X, y):
    LR = sklearn.linear_model.LogisticRegression(C=1, fit_intercept=False)
    LR.fit(X, y)
    LR_p = LR.predict_proba(X)

    print 'sklearn coef={}'.format(LR.coef_)


iris = datasets.load_iris()
X = iris.data[:100, ]
y = np.transpose(iris.target[:100])

eval_my_lr(X, y)
eval_sklearn_lr(X, y)

'''
for i in xrange(0, len(y)):
    pred = my_lr.predict(X[i, :])

    print '{}, {}, {}'.format(y[i], pred, LR_p[i, 1])'''
