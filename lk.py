# Implementation of Lucas-Kanade optical flow.
#
# Specifically this implementation takes greyscale images, minimizes pixel-wise least squares in
# windows, uses a gaussian pyramid, models warps as translations, and iteratively computes additive
# displacements.
#
# This implementation is written with algo_tracking.pdf as a reference.

import numpy as np
import scipy.signal
import scipy.interpolate

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

# Takes an RGB image, im,  with dimensions (X, Y, C) and returns a luminance image of dimensions
# (X, Y). The input image can have any range and type. The returned image will have the same range
# and will be float type.
def ToGreyscale(im):
    # These constants come from Wikipedia.
    return .2126 * im[:, :, 0] + .7152 * im[:, :, 1] + .0722 * im[:, :, 2]

# Takes two luminance images im_0 and im_1 and calculates the optical flow from im_0 to im_1.
def LK(im_0, im_1, window_x = 3, window_y = 3, levels = 3, K = 5, det_thres = .1, v_thres = .01):
    assert(im_0.shape == im_1.shape)
    assert(im_0.shape[0] % 2**(levels - 1) == 0)
    assert(im_0.shape[1] % 2**(levels - 1) == 0)

    pyramid_0 = GaussianPyramid(im, levels)
    pyramid_1 = GaussianPyramid(im, levels)

    # Our warp is parameterized as (Dx, Dy).
    guess_warp = np.zeros((*pyramid_0[-1].shape, 2))

    for level in range(levels - 1, -1, -1):
        I_0 = pyramid_0[level]
        I_1 = pyramid_1[level]

        x_r = np.arange(I_1.shape[0])
        y_r = np.arange(I_1.shape[1])
        # These matrices are used with the griddata() call below to sample our matrix at
        # non-integer points.
        # coords is a matrix such that coords[x, y] = [x, y] and is of dimensions (X, Y, 2)
        # coord_list is the same matrix reshaped to be (XY, 2)
        # value list is I_1 sampled at the coord_list coordiants and is of size (XY,)
        coords = np.stack(np.meshgrid(x_r, y_r, indexing='ij'), axis=-1)
        coord_list = coords.reshape(-1, 2, 1).squeeze()
        value_list = I_1.reshape(-1, 1).squeeze()

        # gradient's dimensions are (X, Y, 2) with gradient[x, y, 0] = dX and gradient[..., 1] = dY
        gradient = np.zeros((*I_0.shape, 2))
        gradient[:, :, 0], gradient[:, :, 1] = np.gradient(I_0)

        # This line gives [dx dy]T * [dx dy] at each (x, y).
        # The resulting tensor has dimensions (X, Y, 2, 2)
        structure_tensor = gradient[:, :, :, np.newaxis] * gradient[:, :, np.newaxis, :]

        G = np.convolve(structure_tensor, np.ones((window_x, window_y, 1, 1)), mode='same')

        # Find G_inv of dimensions (X, Y, 2, 2)
        # We could try using np.linalg.inv here for cleaner code, but catching dealing with
        # singular matrices is messy that way.
        G_det = G[:, :, 0, 0] * G[:, :, 1, 1] - G[:, :, 0, 1] * G[:, :, 1, 0]
        well_posed = spatial_det > det_thres
        inv_G_det = np.where(well_pose, 1/G_det, 0)
        G_inv = np.empty(G_det.shape)
        G_inv[:, :, 0, 0] =  inv_G_det * G[:, :, 1, 1]
        G_inv[:, :, 1, 0] = -inv_G_det * G[:, :, 1, 0]
        G_inv[:, :, 0, 1] = -inv_G_det * G[:, :, 0, 1]
        G_inv[:, :, 1, 1] =  inv_G_det * G[:, :, 0, 0]

        v = np.zeros((*I_0.shape, 2, 1))

        x, y = np.arange(I_0.shape[0]), np.arange(I_0.shape[1])

        for k in range(K)
            # (X, Y, 2)
            warped_coords = coords + guess_warp + v
            # (XY, 2)
            warped_coord_list = warped_coords.reshape(-1, 2, 1).squeeze()

            # Clip coord_list's coordinates with the dimensions of I_1.
            # Note: It might be preferable here to extrapolate values as opposed to clipping our
            # sampling, but the griddata call makes that difficult here.
            warped_coord_list[:, 0] = np.clip(warped_coord_list[:, 0], 0, I_1.shape[0] - 1)
            warped_coord_list[:, 1] = np.clip(warped_coord_list[:, 1], 0, I_1.shape[1] - 1)
            I_1_warped_list = scipy.interpolate.griddata(coord_list, value_list, warped_coords)
            I_1_warped = I_1_warped_list.reshape(*I_0.shape)
            dI = I_0 - I_1_warped

            # (X, Y, 2, 1)
            image_mismatch = dI * gradient[:, :, :, np.newaxis]

            # (X, Y, 2, 2) * (X, Y, 2, 1) = (X, Y, 2, 1)
            dV = G_inv * b
            v = v + dV
            if np.max(np.abs(dV)) < v_thes:
                break

        guess_warp += v
        if level > 0:
            guess_warp = np.zeros((*pyramid_0[level - 1], 2))
            guess_warp[::2, ::2, :] = warp
            guess_warp[1::2, 1::2, :] = (warp + warp[1:, 1:]) / 2
            guess_warp = new_guess_warp
    return warp
