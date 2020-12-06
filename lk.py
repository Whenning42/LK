# Implementation of Lucas-Kanade optical flow.
#
# Specifically this implementation takes greyscale images, minimizes pixel-wise least squares in
# windows, uses a gaussian pyramid, models warps as translations, and iteratively computes additive
# displacements.
#
# This implementation is written with algo_tracking.pdf as a reference.

import numpy as np
import scipy.signal

# Takes in im, an np array of shape (X,Y) as well as an int levels returns a list, of length levels,
# containing a guassian pyramid whose images are dimensions (X, Y), (X/2, Y/2), ...
def GaussianPyramid(im, levels):
    gaussian_1d = np.array([[1/4, 1/2, 1/4]])
    gaussian = gaussian_1d * gaussian_1d.transpose()

    pyramid = []
    last_im = im.copy()
    pyramid.append(last_im)
    for i in range(levels - 1):
        convolved = scipy.signal.convolve2d(last_im, gaussian, mode='same', boundary='symm')
        downsampled = convolved[::2, ::2]
        pyramid.append(downsampled)
        last_im = downsampled

    return pyramid
