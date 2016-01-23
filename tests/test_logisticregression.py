import unittest
import numpy as np
import logisticregression

np.random.seed(seed=1)

def make_test_dataset():
    num_obs = 100000
    params = [0.1, -0.2, 0.3]

    X = np.ndarray(shape=(num_obs, len(params)))
    y = np.ndarray(shape=(num_obs))

    for i in xrange(0, num_obs):
        x = np.random.normal(size=len(params))
        proba = logisticregression.sigmoid(params, x)
        y[i] = np.random.binomial(1, proba , 1)
        X[i, :] = x
    return X, y, params


class TestLogisticRegression(unittest.TestCase):
    def test_auto_step_size(self):
        X, y, true_params = make_test_dataset()

        lr = logisticregression.LogisticRegression(reg_param=0.00, store_iter_loss=False, step_size='auto',
                                                              is_verbose=False)
        lr.fit(X, y)

        rel_error = np.abs((lr.coef - true_params) / true_params)
        self.assertTrue(np.max(rel_error) < 0.1)


if __name__ == '__main__':
    unittest.main()
