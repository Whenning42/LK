import lk
import numpy as np

def TestGaussianPyramid():
    orig = np.zeros((64, 64))
    orig[0:16, 0:16] = 1
    pyramid = lk.GaussianPyramid(orig, 3)

    assert(len(pyramid) == 3)
    assert(np.array_equal(orig, pyramid[0]))

    # We don't put overly tight constraints on the downsampled pyramid values to allow us to change
    # filter sizes without breaking this test.
    assert(pyramid[1].shape == (32, 32))
    assert(pyramid[1][2, 2] > .8)
    assert(pyramid[1][20, 20] == 0)

    assert(pyramid[2].shape == (16, 16))
    assert(pyramid[2][1, 1] > .8)
    assert(pyramid[2][10, 10] == 0)

TestGaussianPyramid()
print("All tests passed!")
