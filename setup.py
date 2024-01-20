#!/usr/bin/env python3
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import re
import subprocess

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("anisoap/__init__.py").read()
).group(1)


class BuildFFI(build_ext):
    def run(self):
        subprocess.run(["make", "-C", "anisoap_rust_lib/"])


if __name__ == "__main__":
    setup(
        version=__version__,
        ext_modules=[
            Extension(name="anisoap", sources=[]),
        ],
        cmdclass={"build_ext": BuildFFI},
        packages=find_packages(exclude=["*__pycache__*"]),
    )
