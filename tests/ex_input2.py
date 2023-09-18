# A temporary test file
import numpy as np
from anisoap.lib import ellipsoid_transform
import pathlib

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

if __name__ == "__main__":
    # path_constructor = lambda file: str(pathlib.Path(__file__).parent.parent.absolute()) + "/benchmarks/two_particle_gb/" + file + ".xyz"
    path_prefix = "./benchmarks/two_particle_gb/"
    ellipsoid_transform(f"{path_prefix}ellipsoid_frames.xyz")