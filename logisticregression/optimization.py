import random

import numpy as np

np.random.seed(seed=1)

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

            if grad_mag < self.min_grad_mag_for_backtracking \
                    or self.step_size < self.min_step_size \
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

    def fit(self, loss_func, loss_grad_func, initial_guess, num_samples):
    # TODO: Unify the declaration of this function in this class and StochasticGradientDescent.
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

            termination = self.termination_criteria(cur_loss / num_samples, prev_loss / num_samples, params_update, it,
                                                    self.max_iter, self.reltol,
                                                    self.abstol)
            params_to_solve -= params_update

            prev_loss = cur_loss
            it += 1

            if self.store_iter_loss:
                self.iter_loss.append(cur_loss)

            if self.is_verbose and np.mod(it, 1) == 0:
                print 'Iteration {}: loss={}. step_size={}.\n  b={}\n, update={}'.format(
                    it, cur_loss / 100000, step_size, params_to_solve, params_update)
        return params_to_solve

    def termination_criteria(self, cur_loss, prev_loss, update, it, max_iter, reltol, abstol):
        termination = False
        abstol_criterion = np.max(np.abs(update)) < abstol
        if cur_loss != 0:
            reltol_criterion = np.abs(prev_loss - cur_loss) / cur_loss < reltol
        else:
            reltol_criterion = False
        if abstol_criterion or reltol_criterion or it >= max_iter:
            termination = True
        return termination


def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

class StochasticGradientDescent:
    PARAMS_MIX_FACTOR = 0.9

    def __init__(self, tol=1e-10, abstol=1e-14, max_iter=10000, step_size_selector=BacktrackingStepSizeSelector(),
                 is_verbose=False, store_iter_loss=False, batch_size=512, num_passes=100):
        self.num_passes = num_passes
        self.batch_size = batch_size
        self.tol = tol
        self.abstol = abstol
        self.max_iter = max_iter
        self.store_iter_loss = store_iter_loss
        self.is_verbose = is_verbose
        self.step_size_selector = step_size_selector

    def fit(self, loss_func, loss_grad_func, initial_guess, X, y):
        for i in xrange(0, self.num_passes):
            obs_indices = range(0, len(y))
            random.shuffle(obs_indices)
            for chunk_num, chunk_ind in enumerate(chunks(obs_indices, self.batch_size)):
                if len(chunk_ind) < self.batch_size:
                    break

                gd = GradientDescent(tol=self.tol, abstol=self.abstol,
                                     max_iter=1, is_verbose=False,
                                     store_iter_loss=False,
                                     step_size_selector=self.step_size_selector)
                X_chunk = X[chunk_ind, :]
                y_chunk = y[chunk_ind]

                fit_result = gd.fit(lambda b : loss_func(X_chunk, y_chunk, b),
                                       lambda b : loss_grad_func(X_chunk, y_chunk, b),
                                       initial_guess, num_samples=X_chunk.shape[0])
                initial_guess = self.PARAMS_MIX_FACTOR * initial_guess + (1 - self.PARAMS_MIX_FACTOR) * fit_result

                if self.is_verbose:
                    print '{} | {} | {}'.format(chunk_num, initial_guess, len(chunk_ind))
        return initial_guess