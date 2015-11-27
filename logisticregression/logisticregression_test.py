# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:23:03 2015

@author: Vic
"""
import sklearn.linear_model

from logisticregression import *
from sklearn import datasets
'''
m = 5
n = 6
X = np.zeros((n,m))
#X[:, 0] = 1.0 * np.ones((5))
#X[:, 1] = -1.0 * np.ones((5))
X[:, 0] = 0.5
X[2:5, 1] = 0.1
y = np.ones((n,1))
y[0:2, :] = 0
b0 = np.zeros((1, m))
'''
iris = datasets.load_iris()
X = iris.data[:100, :3]  # we only take the first two features.
y = np.transpose(iris.target[:100])

my_lr = LogisticRegression(reg_param=0.5)
my_lr.fit(X, y)


print 'analytic b={}'.format(my_lr.coef)

LR = sklearn.linear_model.LogisticRegression(C=1,  fit_intercept=False)
LR.fit(X, y)
LR_p = LR.predict_proba(X)

print 'sklearn coef={}'.format(LR.coef_)

LR_p = LR.predict_proba(X)
'''
for i in xrange(0, len(y)):
    pred = p(b, X[i, :])

    print '{}, {}, {}'.format(y[i], pred, LR_p[i, 1])'''