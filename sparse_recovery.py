import matplotlib.image as mplimg
import numpy as np
from cs_utils import *
from OMP import OMP
from ISTA import ISTA


# S is assumed to be a vector
def sample_y(S, DMD):
    noise_var = 1e-3
    noise = np.random.normal(scale=noise_var, size=len(DMD))

    return DMD @ S + noise


def DMD_matrix(M, N):
    return np.random.choice(a=np.array([1, -1]), size=(M, N))


def CS_matrix(DMD, U_kron_U):
    return DMD @ U_kron_U


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
