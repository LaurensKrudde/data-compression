import numpy as np


def shrinkage_operation(x, eta, lam):
    for i in range(len(x)):
        if x[i] > lam * eta:
            x[i] -= lam * eta
        elif x[i] < lam * eta:
            x[i] += lam * eta
        else:
            x[i] = 0
    return x


def ISTA(y, A, eps):

    eta = 0.01
    lam = 1
    error = 1.0

    M = len(A)
    N = len(A[0])

    z = np.zeros(N)

    while error > eps:

        p = z - 2 * eta * A.T @ (A @ z - y)

        z = shrinkage_operation(p, eta, lam)

        error = np.linalg.norm(y - A @ z, ord=2)**2 + lam * np.linalg.norm(z, ord=1)

    return z