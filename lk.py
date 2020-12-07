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

def StructureTensor(im):
    # gradient's dimensions are (X, Y, 2) with gradient[x, y, 0] = dX and gradient[..., 1] = dY
    gradient = np.zeros((*im.shape, 2))
    gradient[:, :, 0], gradient[:, :, 1] = np.gradient(im)

    # This line gives [dx dy]T * [dx dy] at each (x, y).
    # The resulting tensor has dimensions (X, Y, 2, 2)
    return gradient[:, :, :, np.newaxis] * gradient[:, :, np.newaxis, :]

def SpatialMatrix(structure_tensor, window_x, window_y):
        G = np.empty(structure_tensor.shape)
        G[:, :, 0, 0] = scipy.signal.convolve2d(structure_tensor[:, :, 0, 0], \
                                                np.ones((window_x, window_y)), mode='same')
        G[:, :, 1, 0] = scipy.signal.convolve2d(structure_tensor[:, :, 1, 0], \
                                                np.ones((window_x, window_y)), mode='same')
        G[:, :, 0, 1] = scipy.signal.convolve2d(structure_tensor[:, :, 0, 1], \
                                                np.ones((window_x, window_y)), mode='same')
        G[:, :, 1, 1] = scipy.signal.convolve2d(structure_tensor[:, :, 1, 1], \
                                                np.ones((window_x, window_y)), mode='same')
        return G

def PixelwiseInvert(m, det_thres):
    # We could try using np.linalg.inv here for cleaner code, but catching dealing with
    # singular matrices is messy that way.
    m_det = m[:, :, 0, 0] * m[:, :, 1, 1] - \
            m[:, :, 0, 1] * m[:, :, 1, 0]
    well_posed = np.abs(m_det) > det_thres

    # Setting m_det to 1 where not well posed prevents a warning on the next line when we find
    # 1/m_det
    m_det = np.where(well_posed, m_det, 1)
    inv_m_det = np.where(well_posed, 1/m_det, 0)

    m_inv = np.empty(m.shape)
    m_inv[:, :, 0, 0] =  inv_m_det * m[:, :, 1, 1]
    m_inv[:, :, 1, 0] = -inv_m_det * m[:, :, 1, 0]
    m_inv[:, :, 0, 1] = -inv_m_det * m[:, :, 0, 1]
    m_inv[:, :, 1, 1] =  inv_m_det * m[:, :, 0, 0]
    return m_inv

def BilinearUpsample(im):
    upsampled = np.zeros((im.shape[0] * 2, im.shape[1] * 2, *im.shape[2:]))
    upsampled[0::2,   0::2,   :] = im
    upsampled[1:-2:2, 0::2,   :] = (im[:-1, :]   + im[1:, :]) / 2
    upsampled[0::2,   1:-2:2, :] = (im[:, :-1]   + im[:, 1:]) / 2
    upsampled[1:-2:2, 1:-2:2, :] = (im[:-1, :-1] + im[1:, :-1] + \
                                    im[:-1, 1:]  + im[1:, 1:]) / 4
    return upsampled

# Takes two luminance images im_0 and im_1 and calculates the optical flow from im_0 to im_1.
def LK(im_0, im_1, window_x = 3, window_y = 3, levels = 3, K = 10, det_thres = .1, v_thres = .01):
    assert(im_0.shape == im_1.shape)
    assert(im_0.shape[0] % 2**(levels - 1) == 0)
    assert(im_0.shape[1] % 2**(levels - 1) == 0)

    pyramid_0 = GaussianPyramid(im_0, levels)
    pyramid_1 = GaussianPyramid(im_1, levels)

    # Our warp is parameterized as guess_warp[x, y] = [Dx, Dy].
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
        # value list is I_1 sampled at the coord_list coordinates and is of size (XY,)
        coords = np.stack(np.meshgrid(x_r, y_r, indexing='ij'), axis=-1)
        coord_list = coords.reshape(-1, 2, 1).squeeze()
        value_list = I_1.reshape(-1, 1).squeeze()

        # gradient's dimensions are (X, Y, 2) with gradient[x, y, 0] = dX and gradient[..., 1] = dY
        gradient = np.zeros((*I_0.shape, 2))
        gradient[:, :, 0], gradient[:, :, 1] = np.gradient(I_0)

        structure_tensor = StructureTensor(I_0)
        G = SpatialMatrix(structure_tensor, window_x, window_y)
        G_inv = PixelwiseInvert(G, det_thres = det_thres)

        v = np.zeros((*I_0.shape, 2, 1))

        x, y = np.arange(I_0.shape[0]), np.arange(I_0.shape[1])

        for k in range(K):
            # (X, Y, 2)
            warped_coords = coords + guess_warp + v.squeeze()
            # (XY, 2)
            warped_coord_list = warped_coords.reshape(-1, 2, 1).squeeze()

            # Clip coord_list's coordinates with the dimensions of I_1.
            # Note: It might be preferable here to extrapolate values as opposed to clipping our
            # sampling, but the griddata call makes that difficult here.
            warped_coord_list[:, 0] = np.clip(warped_coord_list[:, 0], 0, I_1.shape[0] - 1)
            warped_coord_list[:, 1] = np.clip(warped_coord_list[:, 1], 0, I_1.shape[1] - 1)
            I_1_warped_list = scipy.interpolate.griddata(coord_list, value_list, warped_coords)
            I_1_warped = I_1_warped_list.reshape(*I_0.shape)
            # (X, Y)
            dI = I_0 - I_1_warped

            # Extend and tile dI to be of shape (X, Y, 2, 1) matching that of gradient.
            dI_reshaped = np.tile(dI[:, :, np.newaxis, np.newaxis], (1, 1, 2, 1))
            image_mismatch = dI_reshaped * gradient[:, :, :, np.newaxis]

            # (X, Y, 2, 2) * (X, Y, 2, 1) = (X, Y, 2, 1)
            dV = np.matmul(G_inv, image_mismatch)
            v = v + dV
            print("Level: %d, Iter: %d, Max update: %f" % (level, k, np.max(np.abs(dV))))
            if np.max(np.abs(dV)) < v_thres:
                break

        guess_warp += v.squeeze()
        if level > 0:
            guess_warp = BilinearUpsample(guess_warp)
    return guess_warp
