import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
import matplotlib.image as mplimg


def read_img_to_grayscale(filename):
    # Load the RGB image: (128, 128, 3) array
    img = mplimg.imread('MESSI_image.jpeg')

    # Convert to grayscale: (128, 128) array
    return rgb2gray(img)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def dctmtx(N):
    return dct(np.eye(N), norm='ortho', axis=0)


def plot_gray(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def img_to_patch_list(img):
    patches = []
    for i in range(4):
        for j in range(4):
            patches.append(img[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32])
    return np.array(patches)


def patch_list_to_img(p):
    return np.block([[p[0], p[1], p[2], p[3]],
                     [p[4], p[5], p[6], p[7]],
                     [p[8], p[9], p[10], p[11]],
                     [p[12], p[13], p[14], p[15]],])
