import sys
import numpy

if sys.platform.find("win") >= 0:
    sys.path.append("c:\Python\DLLs")

from .anisoap_rust_lib import compute_moments_ffi

def compute_moments(dil_mat: numpy.ndarray, gau_cen: numpy.ndarray, max_deg: int) -> numpy.ndarray:
    """
    A simple wrapper around the Rust implementation for better integration with Python.
    """
    return compute_moments_ffi(dil_mat, gau_cen, max_deg);
    