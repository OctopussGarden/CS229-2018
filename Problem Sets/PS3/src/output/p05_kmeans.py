from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

K = 16


# Helper functions
def rgb_img_vectorization(img_matrix):
    """

    Args:
        img_matrix: a list(size: [l,l,3]) represent a square image

    Returns: vectorized numpy array (size : [l**2,3] ) represent the image

    """
    return np.array(img_matrix).reshape(len(img_matrix) ** 2, 3)


# K-means
def k_means_rgbimg(imgpath):
    """
    Compress an image (use K colors instead).
    Args:
        imgpath: a string specifies the image path
        x: unlabelled dataset of size (m,3)
        mu: cluster centroids size (K, 3)

    Returns: updated mu
    """
    # Initialize mu to randomly chosen pixel in the image
    old_img = imread(imgpath)
    mu = np.zeros((K, 3), dtype=int)
    x = rgb_img_vectorization(old_img)

    idx = np.random.randint(0, len(old_img) ** 2, size=K)

    for i in range(K):
        mu[i] = x[idx[i], :]

    m, n = x.shape
    max_iter = 1000

    prev_c = None
    it = 0
    norm_mu_x = np.zeros((m, K))
    c = np.zeros(m, dtype=int)
    c_indi = np.zeros((m, K), dtype=int)
    while it < max_iter and (not (prev_c == c).all()):
        prev_c = c
        for i in range(K):
            norm_mu_x[:, i] = np.linalg.norm(mu[i] - x, axis=1)
        c = np.argmin(norm_mu_x, axis=1)
        for i in range(K):
            c_indi[:, i] = [1 if c_ele == i else 0 for c_ele in c]
            mu[i] = x.T.dot(c_indi[:, i]) / np.sum(c_indi[:, i])
        it += 1
    print(f'Number of iterations:{it}')
    new_img = np.zeros(x.shape, dtype=int)
    for i in range(m):
        new_img[i] = mu[c[i]]
    img_array = new_img.reshape((int(np.sqrt(m + 1)), int(np.sqrt(m + 1)), 3))
    # plt.imshow(img_array.astype(int))
    plt.imshow(img_array)
    plt.show()
    return img_array


# # Initialize mu to randomly chosen pixel in the image
# A = imread('../data/peppers-large.tiff')
# mu = np.zeros((K, 3))
# x_t = rgb_img_vectorization(A)
#
# idx = np.random.randint(0, len(A)**2, size=K)
#
# for i in range(K):
#     mu[i] = x_t[idx[i], :]


B = k_means_rgbimg('../data/peppers-large.tiff')
