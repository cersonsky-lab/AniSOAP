from numpy import array
from numpy.testing import assert_allclose


def assert_close(a, b, rtol=1e-10, atol=5e-16):
    a_array = array([a])
    b_array = array([b])
    assert_allclose(a_array, b_array, rtol=rtol, atol=atol)
