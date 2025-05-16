import numpy as np
from anisoap.utils import ClebschGordanReal
from time import perf_counter


class TestCGRCache:
    def test_cgr(self):
        """
        Tests that the CGR cache is correct, and that using the CGR cache results
        in a performance boost.
        """
        lmax = 5
        start = perf_counter()
        mycg1 = ClebschGordanReal(lmax)
        end = perf_counter()
        time_nocache = end - start
        start = perf_counter()
        mycg2 = ClebschGordanReal(lmax)  # we should expect these to be cached!
        end = perf_counter()
        time_cache = end - start
        assert time_cache < time_nocache
        assert mycg1.get_cg().keys() == mycg2.get_cg().keys()
        for key1, key2 in zip(mycg1.get_cg(), mycg2.get_cg()):
            for entry1, entry2 in zip(mycg1.get_cg()[key1], mycg2.get_cg()[key2]):
                assert np.all(entry1 == entry2)
