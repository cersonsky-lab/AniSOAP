import timeit

import numpy as np
import pytest
from anisoap_rust_lib import compute_moments
from scipy.spatial.transform import Rotation as R

from anisoap.utils.moment_generator import compute_moments_inefficient_implementation

rng = np.random.default_rng(12345)

TEST_CEN = rng.uniform(-30, 30, size=(5, 3))
TEST_ROT = R.random(5, rng).as_matrix()
TEST_SEMIAX = 10 ** rng.uniform(-2, 2, size=(5, 3))
TEST_MAT = [ROT @ np.diag(SEMIAX) @ ROT.T for ROT in TEST_ROT for SEMIAX in TEST_SEMIAX]
TEST_LMAX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestComputeMoments:
    @pytest.mark.parametrize("cen", TEST_CEN)
    @pytest.mark.parametrize("mat", TEST_MAT)
    @pytest.mark.parametrize("lmax", TEST_LMAX)
    def test_fixed_input(self, cen, mat, lmax):
        res_ori: np.ndarray[np.float64] = compute_moments_inefficient_implementation(
            mat, cen, lmax
        )
        res_ffi: np.ndarray[np.float64] = compute_moments(mat, cen, lmax)
        assert np.allclose(res_ori, res_ffi)

    @pytest.mark.parametrize("cen", TEST_CEN)
    @pytest.mark.parametrize("mat", TEST_MAT)
    @pytest.mark.parametrize("lmax", TEST_LMAX)
    def test_speedup(self, cen, mat, lmax):
        # Times the average time for function call over num_iter iterations.
        # Prime the compute_moments rust function, since initial startup can be long.
        compute_moments(mat, cen, lmax)
        start = timeit.default_timer()
        num_iter = 3
        for i in range(num_iter):
            compute_moments_inefficient_implementation(mat, cen, lmax)
        time_ori = (timeit.default_timer() - start) / num_iter

        start = timeit.default_timer()
        for i in range(num_iter):
            compute_moments(mat, cen, lmax)
        time_ffi = (timeit.default_timer() - start) / num_iter

        assert time_ffi < time_ori
