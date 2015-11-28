def numeric_derivative(f, x, coord, eps=10e-12):
    x_fw = x.copy()
    x_fw[0, coord] += eps

    x_bw = x.copy()
    x_bw[0, coord] -= eps

    return (f(x_fw) - f(x_bw)) / (2 * eps)

def gradient_descent_numeric(f, b0, step_size=0.01, max_iter=1000):
    b = b0.copy()
    for i in xrange(0, max_iter):
        for j in xrange(0, b0.size):
            #print numeric_derivative(f, b, j)
            b[0, j] = b[0, j] - step_size * numeric_derivative(f, b, j)

        #if np.mod(i, 100) == 0:
        #    print 'Iteration {}: loss={}'.format(i, f(b))
    return b

def gradient_descent(loss_func, grad_func, b0, step_size=0.1, max_iter=100000, tol=0.001):
    b = b0.copy()
    termination = False
    it = 0
    while not termination:
        update = step_size *grad_func(b)

        if np.max(update / b) < tol or it >= max_iter:
            print "Terminated on iter %d" % it
            termination = True

        b = b - update

        it += 1

        if np.mod(it, 100) == 0:
            print 'Iteration {}: loss={}. b={}'.format(it, loss_func(b), b)
    return b