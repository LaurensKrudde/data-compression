import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
from cs_utils import *


def K_sparse_approx_using_id(S, K):

    # Sparsify using identity (does nothing)
    X = S

    # For K-sparse approximation, take the K largest values
    indices = np.argpartition(abs(X), -K)[-K:]
    values = X[indices]

    # Make sparse matrix by setting all values to 0 except the K largest values
    X_sparse = np.zeros(X.shape)
    X_sparse[indices] = values

    # Reconstructing (does nothing)
    return X_sparse


def K_sparse_approx_using_2ddct(S, K):

    # Sparsify using 2D-DCT transform
    X = U_kron_U @ S

    # For K-sparse approximation, take the K largest values
    indices = np.argpartition(abs(X), -K)[-K:]
    values = X[indices]

    # Make sparse matrix by setting all values to 0 except the K largest values
    X_sparse = np.zeros(X.shape)
    X_sparse[indices] = values

    # Reconstruct S from K-sparse approximation
    return U_kron_U_inv @ X_sparse


def MSE_using_id(S, K):
    S_reconstructed = K_sparse_approx_using_id(S, K)
    return np.mean(np.square(S_reconstructed - S))


def MSE_using_2DDCT(S, K):
    S_reconstructed = K_sparse_approx_using_2ddct(S, K)
    return np.mean(np.square(S_reconstructed - S))


if __name__ == '__main__':

    # Load the (128, 128) grayscale array
    img_gray = read_img_to_grayscale('MESSI_image.jpeg')

    # Divide in patches
    patch_list = img_to_patch_list(img_gray)

    # 2D-DCT transformation
    U = dctmtx(32)
    U_kron_U = np.kron(U, U)
    U_kron_U_inv = np.linalg.inv(U_kron_U)

    MSE_id = []
    MSE_2ddct = []

    K_range = np.arange(1, 1024, 8)
    for K in K_range:

        error_id = 0
        error_2ddct = 0

        for patch in patch_list:

            # MSE error
            S = patch.flatten()
            error_id += MSE_using_id(S, K)
            error_2ddct += MSE_using_2DDCT(S, K)

        MSE_id.append(error_id)
        MSE_2ddct.append(error_2ddct)

    plt.plot(K_range, MSE_id, label='Id')
    plt.plot(K_range, MSE_2ddct, label='2D-DCT')
    plt.xlabel('K')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    plt.plot(K_range, np.log10(MSE_id), label='Id')
    plt.plot(K_range, np.log10(MSE_2ddct), label='2D-DCT')
    plt.xlabel('K')
    plt.ylabel('MSE (dB)')
    plt.legend()
    plt.show()
