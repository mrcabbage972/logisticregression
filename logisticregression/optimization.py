import numpy as np


class GradientDescent:
    def __init__(self, tol=1e-10, abstol=1e-14, alpha=0.3, beta=0.5, initial_step_size=2.0, max_iter=10000):
        self.tol = tol
        self.abstol = abstol
        self.alpha = alpha
        self.beta = beta
        self.initial_step_size = initial_step_size
        self.max_iter = max_iter

    def fit(self, loss_func, grad_func, b0):
        b = b0.copy()
        termination = False
        it = 0

        step_size = self.initial_step_size

        prev_loss = np.inf
        while not termination:
            step_size = self.calc_step_size(loss_func, grad_func, b, step_size)

            cur_loss = loss_func(b)
            update = step_size * grad_func(b)

            termination = self.termination_criteria(cur_loss, prev_loss, update, it, self.max_iter, self.tol,
                                                    self.abstol)

            b -= update

            prev_loss = cur_loss
            it += 1

            if np.mod(it, 1) == 0:
                print 'Iteration {}: loss={}. step_size={}.\n  b={}\n, update={}'.format(it, cur_loss, step_size, b,
                                                                                         update)
        return b

    def termination_criteria(self, cur_loss, prev_loss, update, it, max_iter, tol, abstol):
        termination = False
        abstol_criterion = np.max(np.abs(update)) < abstol
        reltol_criterion = np.max(prev_loss / (cur_loss + 1e-10)) < tol
        if abstol_criterion or reltol_criterion or it >= max_iter:
            print "Terminated on iter %d" % it
            termination = True
        return termination

    def calc_step_size(self, loss_func, grad_func, b, step_size):
        alpha = 0.3
        beta = 0.5

        step_size_criteria = False
        step_size /= beta ** 2
        while not step_size_criteria:
            step_size *= beta
            cur_grad = grad_func(b)
            cur_loss = loss_func(b)
            lhs = loss_func(b - step_size * grad_func(b))
            grad_mag = np.sum(cur_grad * cur_grad)
            if grad_mag < 1e-10:
                step_size_criteria = True
            else:
                rhs = cur_loss - alpha * step_size * grad_mag
                step_size_criteria = lhs <= rhs

        return step_size
