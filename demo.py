import numpy as np
from logisticregression import LogisticRegression

X = np.array([[ 4.6, 3.2, 1.4, 0.2],
 [ 5.3, 3.7, 1.5, 0.2],
 [ 5.,  3.3, 1.4, 0.2],
 [ 7.,  3.2, 4.7, 1.4],
 [ 6.4, 3.2, 4.5, 1.5],
 [ 6.9, 3.1, 4.9, 1.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lr = LogisticRegression(reg_param=0.01)
lr.fit(X, y)
pred = lr.predict(X)

mse = np.sum((pred - y)**2)

print 'Mean squared error=%f' % mse

for i in xrange(0, len(y)):
    print '{}, {}'.format(y[i], pred[i])