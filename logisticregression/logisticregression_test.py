# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:23:03 2015

@author: Vic
"""
import sklearn.linear_model

from logisticregression import *
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:100, ]  # we only take the first two features.
y = np.transpose(iris.target[:100])

my_lr = LogisticRegression(reg_param=0.5)
my_lr.fit(X, y)


print 'my coef={}'.format(my_lr.coef)

LR = sklearn.linear_model.LogisticRegression(C=1,  fit_intercept=False)
LR.fit(X, y)
LR_p = LR.predict_proba(X)

print 'sklearn coef={}'.format(LR.coef_)

LR_p = LR.predict_proba(X)

'''
for i in xrange(0, len(y)):
    pred = my_lr.predict(X[i, :])

    print '{}, {}, {}'.format(y[i], pred, LR_p[i, 1])'''