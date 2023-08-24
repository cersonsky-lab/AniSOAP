#!/usr/bin/env python3
from setuptools import setup
import re
import subprocess

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("anisoap/__init__.py").read()
).group(1)

if __name__ == "__main__":
    setup(version=__version__)
    subprocess.run(["cd", "anisoap_rust_lib"])
    subprocess.run(["make"])
    subprocess.run(["cd", "../"])
