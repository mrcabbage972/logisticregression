import numpy as np


class FixedStepSizeSelector:
    def __init__(self, step_size):
        self.step_size = step_size

    def calc_step_size(self, loss_func, grad_func, cur_loss, cur_grad, params_to_solve):
        return self.step_size

class BacktrackingStepSizeSelector:
    def __init__(self, step_size_min_reduction_relative_to_grad=0.3, step_size_reduction_factor=0.5,
                 initial_step_size=2.0):
        self.step_size = initial_step_size
        self.step_size_min_reduction_relative_to_grad = step_size_min_reduction_relative_to_grad
        self.step_size_reduction_factor = step_size_reduction_factor
        self.min_grad_mag_for_backtracking = 1e-10
        self.min_step_size = 1e-10
        self.max_step_size = 100

    def calc_step_size(self, loss_func, grad_func, cur_loss, cur_grad, params_to_solve):
        step_size_criteria = False
        self.step_size /= self.step_size_reduction_factor ** 2
        grad_mag = np.sum(cur_grad * cur_grad)
        while not step_size_criteria:
            self.step_size *= self.step_size_reduction_factor
            lhs = loss_func(params_to_solve - self.step_size * grad_func(params_to_solve))

            if grad_mag < self.min_grad_mag_for_backtracking\
                    or self.step_size < self.min_step_size\
                    or self.step_size > self.max_step_size:
                self.step_size = np.maximum(self.min_step_size, np.minimum(self.max_step_size, self.step_size))
                step_size_criteria = True
            else:
                rhs = cur_loss - self.step_size_min_reduction_relative_to_grad * self.step_size * grad_mag
                step_size_criteria = lhs <= rhs

        return self.step_size

class GradientDescent:
    def __init__(self, tol=1e-10, abstol=1e-14, max_iter=10000, step_size_selector=BacktrackingStepSizeSelector(),
                 is_verbose=False, store_iter_loss=False):
        self.step_size_selector = step_size_selector
        self.store_iter_loss = store_iter_loss
        self.is_verbose = is_verbose

        # Termination criteria
        self.reltol = tol
        self.abstol = abstol
        self.max_iter = max_iter

        if self.store_iter_loss:
            self.iter_loss = []
        else:
            self.iter_loss = None

    def fit(self, loss_func, loss_grad_func, initial_guess):
        params_to_solve = initial_guess.copy()
        termination = False
        it = 0

        prev_loss = np.inf
        while not termination:
            cur_loss = loss_func(params_to_solve)
            cur_grad = loss_grad_func(params_to_solve)

            step_size = self.step_size_selector.calc_step_size(
                loss_func, loss_grad_func, cur_loss, cur_grad, params_to_solve)

            cur_loss = loss_func(params_to_solve)
            params_update = step_size * loss_grad_func(params_to_solve)

            termination = self.termination_criteria(cur_loss, prev_loss, params_update, it, self.max_iter, self.reltol,
                                                    self.abstol)

            params_to_solve -= params_update

            prev_loss = cur_loss
            it += 1

            if self.store_iter_loss:
                self.iter_loss.append(cur_loss)

            if self.is_verbose and np.mod(it, 1) == 0:
                print 'Iteration {}: loss={}. step_size={}.\n  b={}\n, update={}'.format(
                    it, cur_loss, step_size, params_to_solve, params_update)
        return params_to_solve

    def termination_criteria(self, cur_loss, prev_loss, update, it, max_iter, reltol, abstol):
        termination = False
        abstol_criterion = np.max(np.abs(update)) < abstol
        reltol_criterion = np.max(prev_loss / (cur_loss + 1e-10)) < reltol
        if abstol_criterion or reltol_criterion or it >= max_iter:
            termination = True
        return termination