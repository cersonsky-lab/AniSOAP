import sys

if sys.platform.find("win") >= 0:
    sys.path.append("c:\Python\DLLs")

from .anisoap_rust_lib import compute_moments
