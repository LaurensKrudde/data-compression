import matplotlib.image as mplimg
import numpy as np
from cs_utils import *


# S is assumed to be a vector
def sample_y(S, DMD):
    noise_var = 1e-3
    noise = np.random.normal(scale=noise_var, size=len(DMD))

    return DMD @ S + noise


def DMD_matrix(M, N):
    return np.random.choice(a=np.array([1, -1]), size=(M, N))


def CS_matrix(DMD, U_kron_U):
    return DMD @ U_kron_U


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

    print(f"Stopped after {i} iterations.")
    return z


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


if __name__ == '__main__':

    # Load the (128, 128) grayscale array
    img_gray = read_img_to_grayscale('MESSI_image.jpeg')

    # Take one patch
    S_patch = img_gray[32:64, 32:64]
    plot_gray(S_patch)

    # Vectorize
    S_vec = S_patch.flatten()
    N = S_vec.size

    # Get the sparsifying matrix U
    U = dctmtx(32)
    U_kron_U = np.kron(U, U)
    U_kron_U_inv = np.linalg.inv(U_kron_U)

    # Number of samples
    M = 128

    # DMD matrix
    DMD = DMD_matrix(M, N)

    # Get M measurements
    y = sample_y(S_vec, DMD)

    # Get the CS matrix A
    A = CS_matrix(DMD, U_kron_U)

    # Estimate of x using OMP
    x_hat = OMP(y, A, 1e-3)

    # Estimate of x using ISTA
    # x_hat = ISTA(y, A, 1e-3)

    # Reconstruct S
    S = U_kron_U_inv @ x_hat

    plot_gray(S.reshape(32,32))
