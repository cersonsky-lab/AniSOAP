import numpy as np
from ase import Atoms
from time import perf_counter


a = 3
b = c = 1

def generate_frame(l1, l2, l3):
    n_tot = l1 * l2 * l3
    positions = np.asarray(
        [[i, j, 3 * k] for i in range(l1) for j in range(l2) for k in range(l3)]
    )

    frame = Atoms(positions=positions, numbers=[0] * n_tot)
    frame.arrays["quaternions"] = np.array([[1, 0, 0, 0]] * n_tot)
    frame.arrays[r"c_diameter\[1\]"] = a * np.ones(n_tot)
    frame.arrays[r"c_diameter\[2\]"] = b * np.ones(n_tot)
    frame.arrays[r"c_diameter\[3\]"] = c * np.ones(n_tot)

    frame.arrays[r"c_diameter[1]"] = a * np.ones(n_tot)
    frame.arrays[r"c_diameter[2]"] = b * np.ones(n_tot)
    frame.arrays[r"c_diameter[3]"] = c * np.ones(n_tot)

    return [frame]


class TestCGRCache:

    def test_cgr(self):
        pass 