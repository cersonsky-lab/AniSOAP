#!/usr/bin/env python3
from setuptools import setup
from setuptools.command.build_ext import build_ext
import re
import subprocess
import sys


__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("anisoap/__init__.py").read()
).group(1)


class BuildFFI(build_ext):
    def run(self):
        if subprocess.call(["cd", "anisoap_rust_lib"]) != 0:
            print("cd into rust library folder was not successful.")
            sys.exit(-1)
        if subprocess.call(["make"]) != 0:
            print("makefile in rust library folder was not successful.")
            sys.exit(-1)
        if subprocess.call(["cd", "anisoap_rust_lib"]) != 0:
            print("cd out of the rust library folder was not successful.")
            sys.exit(-1)

        super().run()


if __name__ == "__main__":
    setup(version=__version__, cmdclass={'build_ext': BuildFFI})
