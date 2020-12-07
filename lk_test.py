import lk
import numpy as np
import math

def TestGaussianPyramid():
    orig = np.zeros((64, 64))
    orig[0:16, 0:16] = 1
    pyramid = lk.GaussianPyramid(orig, 3)

    assert(len(pyramid) == 3)
    assert(np.array_equal(orig, pyramid[0]))

    # We don't put overly tight constraints on the expected pyramid values to allow us to change
    # filter sizes without breaking this test.
    assert(pyramid[1].shape == (32, 32))
    assert(pyramid[1][2, 2] == 1)
    assert(pyramid[1][20, 20] == 0)

    assert(pyramid[2].shape == (16, 16))
    assert(pyramid[2][1, 1] == 1)
    assert(pyramid[2][10, 10] == 0)

def TestToGreyscale():
    im = np.zeros((4, 4, 3))
    im[0, 0, :] = 1
    im[0, 1, :] = 0
    greyscale = lk.ToGreyscale(im)

    assert(greyscale.shape == (4, 4))
    assert(math.isclose(greyscale[0, 0], 1))
    assert(math.isclose(greyscale[0, 1], 0))

# Skip for now
def TestLinearInterpolation():
    pass

def TestStructureTensor():
    dx = 1
    dy = -2
    a = np.zeros((2, 2))
    a[0, 1] = dx
    a[1, 0] = dy
    st = lk.StructureTensor(a)

    assert(math.isclose(st[0, 0, 0, 0], dx ** 2))
    assert(math.isclose(st[0, 0, 1, 0], dx * dy))
    assert(math.isclose(st[0, 0, 0, 1], dx * dy))
    assert(math.isclose(st[0, 0, 1, 1], dy ** 2))

def TestSpatialMatrix():
    a = np.zeros((3, 5))
    a[:, :] += np.tile(np.arange(3)[:, np.newaxis], (1, 5)) ** 2
    a[:, :] += np.tile(np.arange(5)[np.newaxis, :], (3, 1))
    st = lk.StructureTensor(a)
    sm = lk.SpatialMatrix(st, window_x = 3, window_y = 3)

    assert(np.allclose(sm[0, 0], st[0, 0] + st[1, 0] + \
                                 st[0, 1] + st[1, 1]))

    assert(np.allclose(sm[1, 0],           sm[0, 0] + st[2, 0] \
                                                    + st[2, 1]))

    assert(np.allclose(sm[1, 1],            sm[1, 0] \
                               + st[0, 2] + st[1, 2] + st[2, 2]))

def TestPixelwiseInversion():
    matrices = np.zeros((2, 1, 2, 2))
    # Invertible
    m_0 = np.array([[0, 1], [-1, 0]])
    # Singular
    m_1 = np.array([[1, 1], [1, 1]])
    matrices[0, 0] = m_0
    matrices[1, 0] = m_1

    inverted = lk.PixelwiseInversion(matrices)

    assert(inverted.shape == matrices.shape)
    assert(np.allclose(m_0 * inverted[0, 0], np.eye(2)))
    assert(np.allclose(m_1, np.zeros(2, 2)))

def TestBilinearUpsample():
    a = np.zeros((2, 2, 2))
    a[0, 0] = [-1, 2]
    a[0, 1] = [-2, 4]
    a[1, 0] = [-3, 6]
    a[1, 1] = [-4, 8]

    b = lk.BilinearUpsample(a)

    assert(b.shape[0] == a.shape[0] * 2)
    assert(b.shape[1] == a.shape[1] * 2)

    assert(np.allclose(b[0, 0], a[0, 0]))
    assert(np.allclose(b[0, 2], a[0, 1]))
    assert(np.allclose(b[2, 0], a[1, 0]))
    assert(np.allclose(b[2, 2], a[1, 1]))
    assert(np.allclose(b[1, 0], (a[0, 0] + a[1, 0]) / 2))
    assert(np.allclose(b[0, 1], (a[0, 0] + a[0, 1]) / 2))
    assert(np.allclose(b[1, 1], (a[0, 0] + a[1, 0] + \
                                 a[0, 1] + a[1, 1]) / 4))

    a = np.zeros((3, 3, 2))
    b = lk.BilinearUpsample(a)
    assert(b.shape[0] == a.shape[0] * 2)
    assert(b.shape[1] == a.shape[1] * 2)

def TestLKSimple():
    pass

TestGaussianPyramid()
TestToGreyscale()
TestBilinearUpsample()
print("All tests passed!")
