import numpy
import pytest
from anisoap_rust_lib import compute_moments
from anisoap.utils.moment_generator import compute_moments_inefficient_implementation
import timeit


TEST_CEN = [
    numpy.array([1.0, 2.0, 3.0]),
    numpy.array([0.3, 0.7, 0.1]),
    numpy.array([0.0, 1.0, 0.0]),
    numpy.array([1.3, 5.2, 2.7]),
]
TEST_MAT = [
    numpy.array([[0.81, 0.92, 0.93], [0.92, 0.29, 0.66], [0.93, 0.66, 0.13]]),
    numpy.array([[0.25, 0.54, 0.41], [0.54, 0.79, 0.68], [0.41, 0.68, 0.25]]),
    numpy.array([[0.95, 0.91, 0.22], [0.91, 0.92, 0.23], [0.22, 0.23, 0.70]]),
]
TEST_LMAX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestComputeMoments:
    @pytest.mark.parametrize("cen", TEST_CEN)
    @pytest.mark.parametrize("mat", TEST_MAT)
    @pytest.mark.parametrize("lmax", TEST_LMAX)
    def test_fixed_input(self, cen, mat, lmax):
        res_ori: numpy.ndarray[numpy.float64] = (
            compute_moments_inefficient_implementation(mat, cen, lmax)
        )
        res_ffi: numpy.ndarray[numpy.float64] = compute_moments(mat, cen, lmax)
        assert numpy.allclose(res_ffi, res_ori, 1e-8)

    @pytest.mark.parametrize("cen", TEST_CEN)
    @pytest.mark.parametrize("mat", TEST_MAT)
    @pytest.mark.parametrize("lmax", TEST_LMAX)
    def test_speedup(self, cen, mat, lmax):
        # Times the average time for function call over num_iter iterations.
        # Prime the compute_moments rust function, since initial startup can be long.
        compute_moments(mat, cen, lmax)
        start = timeit.default_timer()
        num_iter = 15
        for i in range(num_iter):
            compute_moments_inefficient_implementation(mat, cen, lmax)
        time_ori = (timeit.default_timer() - start) / num_iter

        start = timeit.default_timer()
        for i in range(num_iter):
            compute_moments(mat, cen, lmax)
        time_ffi = (timeit.default_timer() - start) / num_iter

        assert time_ffi < time_ori

    def test_random_inputs(self):
        rem_tests = 10_000

        while rem_tests > 0:
            # Generates a random number for l_max between 1 and 10.
            rand_lmax = numpy.random.randint(1, 11)

            # Generates 3D vector, each element between -5 and 5.
            random_cen = numpy.random.random((3,)) * 2 * 5 - 5

            # Generates a 3 x 3 matrix and make it positive definite
            random_mat = numpy.random.random((3, 3)) * 2 * 10 - 10
            random_mat = random_mat @ random_mat.T

            rand_mat_det = numpy.linalg.det(random_mat)
            if rand_mat_det < 1e-14:
                continue
            else:
                res_ori: numpy.ndarray[numpy.float64] = (
                    compute_moments_inefficient_implementation(
                        random_mat, random_cen, rand_lmax
                    )
                )
                res_ffi: numpy.ndarray[numpy.float64] = compute_moments(
                    random_mat, random_cen, rand_lmax
                )

                rem_tests -= 1
                # match up to 4 decimal places (in sci. notation)
                assert numpy.allclose(res_ori, res_ffi, rtol=1e-4)
