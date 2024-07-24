#!/usr/bin/env python3
import re

from setuptools import setup

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("anisoap/__init__.py").read()
).group(1)

if __name__ == "__main__":
    setup(version=__version__)
