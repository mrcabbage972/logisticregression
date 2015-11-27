import numpy as np


def gradient_descent_backtrack(loss_func, grad_func, b0, max_iter=10000, tol=1e-10, abstol=1e-14):
    b = b0.copy()
    termination = False
    it = 0

    step_size = 2.0

    prev_loss = np.inf
    while not termination:

        step_size = calc_step_size(loss_func, grad_func, b, step_size)

        cur_loss = loss_func(b)
        update = step_size * grad_func(b)

        termination = termination_criteria(cur_loss, prev_loss, update, it, max_iter, tol, abstol)

        b = b - update

        prev_loss = cur_loss
        it += 1

        if np.mod(it, 1) == 0:
            print 'Iteration {}: loss={}. step_size={}.\n  b={}\n, update={}'.format(it, cur_loss, step_size, b, update)
    return b


def termination_criteria(cur_loss, prev_loss, update, it, max_iter, tol, abstol):
    termination = False
    abstol_criterion = np.max(np.abs(update)) < abstol
    reltol_criterion = np.max(prev_loss / (cur_loss + 1e-10)) < tol
    if abstol_criterion or reltol_criterion or it >= max_iter:
        print "Terminated on iter %d" % it
        termination = True
    return termination


def calc_step_size(loss_func, grad_func, b, step_size):
    alpha = 0.3
    beta = 0.5

    step_size_criteria = False
    step_size = step_size / (beta ** 2)
    while not step_size_criteria:
        step_size = beta * step_size
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
