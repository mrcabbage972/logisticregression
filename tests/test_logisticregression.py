import unittest
import numpy as np
import logisticregression

np.random.seed(seed=1)

def make_test_dataset():
    num_obs = 5000
    params = [0.1, 0.2, 0.01]

    X = np.ndarray(shape=(num_obs, len(params)))
    y = np.ndarray(shape=(num_obs))

    for i in xrange(0, num_obs):
        x = np.random.normal(size=len(params))
        proba = logisticregression.sigmoid(params, x)
        y[i] = np.random.binomial(1, proba , 1)
        X[i, :] = x
    return X, y


class TestLogisticRegression(unittest.TestCase):
    def test_fixed_step_size(self):
        X, y = make_test_dataset()

        lr_fixed_step = logisticregression.LogisticRegression(reg_param=0.00, store_iter_loss=False, step_size=0.001)
        lr_fixed_step.fit(X, y)


if __name__ == '__main__':
    unittest.main()
