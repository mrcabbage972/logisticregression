import numpy as np
from logisticregression import LogisticRegression
import matplotlib.pylab as plt

X = np.array([[ 4.6, 3.2, 1.4, 0.2],
 [ 5.3, 3.7, 1.5, 0.2],
 [ 5.,  3.3, 1.4, 0.2],
 [ 7.,  3.2, 4.7, 1.4],
 [ 6.4, 3.2, 4.5, 1.5],
 [ 6.9, 3.1, 4.9, 1.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lr_fixed_step = LogisticRegression(reg_param=0.01, store_iter_loss=True, step_size=0.01)
lr_fixed_step.fit(X, y)

lr_auto_step = LogisticRegression(reg_param=0.01, store_iter_loss=True, step_size='auto')
lr_auto_step.fit(X, y)


plt.plot(np.arange(0, 100), lr_fixed_step.iter_loss[0:100], 'r', lr_auto_step.iter_loss[0:100], 'g')
plt.legend(['Step size - 0.01', 'Step size - auto'])
plt.show()

#print '%6s %6s' % ('true', 'predicted')
#for i in xrange(0, len(y)):
#    print '%6d %6.4f' %(y[i], pred[i])