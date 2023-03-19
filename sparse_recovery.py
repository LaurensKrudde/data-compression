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


# Not used
def random_measurements(S, M):
    N = len(S)

    DMD = DMD_matrix(N, N)
    noise = np.random.normal(scale=1e-3, size=N)

    y = DMD @ S + noise
    A = DMD @ U_kron_U

    idx = np.random.choice(len(y), M, replace=False)
    y_meas = y[idx]
    A_meas = A[idx, :]

    return A_meas, y_meas


def DMD_matrix(M, N):
    return np.random.choice(a=np.array([1, -1]), size=(M, N))


def CS_matrix(DMD, U_kron_U):
    return DMD @ U_kron_U


if __name__ == '__main__':

    # Load the (128, 128) grayscale array
    img_gray = read_img_to_grayscale('MESSI_image.jpeg')
    img_reconstructed_OMP = []
    img_reconstructed_ISTA = []

    # Get the sparsifying matrix U
    U = dctmtx(32)
    U_kron_U = np.kron(U, U)
    U_kron_U_inv = np.linalg.inv(U_kron_U)

    # Number of samples
    M = 204
    N = 1024

    # DMD matrix
    DMD = DMD_matrix(M, N)

    # Get the CS matrix A
    A = CS_matrix(DMD, U_kron_U_inv)

    for S_patch in img_to_patch_list(img_gray):

        # Vectorize
        S_vec = S_patch.flatten()

        # Get M measurements
        y = sample_y(S_vec, DMD)

        # Estimate of x using OMP
        x_hat_OMP = OMP(y, A, 1e-3)

        # Estimate of x using ISTA
        # x_hat_ISTA = ISTA(y, A, 1e-3)

        # Reconstruct S
        S_recon_OMP = U_kron_U @ x_hat_OMP
        # S_recon_ISTA = U_kron_U @ x_hat_ISTA

        img_reconstructed_OMP.append(S_recon_OMP.reshape((32,32)))
        # img_reconstructed_ISTA.append(S_recon_ISTA)

    plt.imshow(patch_list_to_img(img_reconstructed_OMP), cmap='gray', vmin=0, vmax=255)
    plt.title("Messi reconstructed using OMP with " + str(M) + " samples.")
    plt.show()

    # plt.imshow(patch_list_to_img(img_reconstructed_ISTA), cmap='gray', vmin=0, vmax=255)
    # plt.title("Messi reconstructed using ISTA with " + str(M) + "samples")
    # plt.show()
