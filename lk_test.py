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

TestGaussianPyramid()
TestToGreyscale()
print("All tests passed!")
