import numpy as np


def OMP(y, A, eps):
    N = np.size(A, 1)
    M = len(y)

    z = np.zeros(N)
    Lambda_set = set()
    r = y

    i = 0

    while np.linalg.norm(r, 2) > eps and i < M:

        # Matching step
        h = np.transpose(A) @ r

        # Support identification step
        k = np.argmax(h)

        # Support augmentation step
        Lambda_set.add(k)

        # Update estimate of the sparse vector
        A_Lambda = A[:, list(Lambda_set)]
        z_Lambda = np.linalg.inv(A_Lambda.T @ A_Lambda) @ A_Lambda.T @ y
        z[list(Lambda_set)] = z_Lambda

        # Update residue
        r = y - A @ z

        i += 1

    print(f"OMP stopped after {i} iterations.")
    return z
