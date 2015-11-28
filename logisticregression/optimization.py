import numpy as np


class GradientDescent:
    def __init__(self, tol=1e-10, abstol=1e-14, alpha=0.3, beta=0.5, initial_step_size=2.0, max_iter=10000, is_verbose=False):
        self.reltol = tol
        self.abstol = abstol
        self.alpha = alpha
        self.beta = beta
        self.initial_step_size = initial_step_size
        self.max_iter = max_iter
        self.is_verbose = is_verbose

    def fit(self, loss_func, loss_grad_func, initial_guess):
        params = initial_guess.copy()
        termination = False
        it = 0

        step_size = self.initial_step_size

        prev_loss = np.inf
        while not termination:
            cur_loss = loss_func(params)
            cur_grad = loss_grad_func(params)

            step_size = self.calc_step_size(loss_func, loss_grad_func, cur_loss, cur_grad, params, step_size)

            cur_loss = loss_func(params)
            update = step_size * loss_grad_func(params)

            termination = self.termination_criteria(cur_loss, prev_loss, update, it, self.max_iter, self.reltol,
                                                    self.abstol)

            params -= update

            prev_loss = cur_loss
            it += 1

            if self.is_verbose and np.mod(it, 1) == 0:
                print 'Iteration {}: loss={}. step_size={}.\n  b={}\n, update={}'.format(it, cur_loss, step_size, params,
                                                                                         update)
        return params

    def termination_criteria(self, cur_loss, prev_loss, update, it, max_iter, reltol, abstol):
        termination = False
        abstol_criterion = np.max(np.abs(update)) < abstol
        reltol_criterion = np.max(prev_loss / (cur_loss + 1e-10)) < reltol
        if abstol_criterion or reltol_criterion or it >= max_iter:
            termination = True
        return termination

    def calc_step_size(self, loss_func, grad_func, cur_loss, cur_grad, b, step_size):
        step_size_criteria = False
        step_size /= self.beta ** 2
        grad_mag = np.sum(cur_grad * cur_grad)
        while not step_size_criteria:
            step_size *= self.beta
            lhs = loss_func(b - step_size * grad_func(b))

            if grad_mag < 1e-10:
                step_size_criteria = True
            else:
                rhs = cur_loss - self.alpha * step_size * grad_mag
                step_size_criteria = lhs <= rhs

        return step_size
