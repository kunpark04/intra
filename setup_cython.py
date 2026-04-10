"""
setup_cython.py — Build the Cython AOT extension for the backtest core.

Usage:
    python setup_cython.py build_ext --inplace

The compiled .pyd (Windows) / .so (Linux/macOS) lands in src/cython_ext/
so that `from src.cython_ext.backtest_core import backtest_core_cy` works
from the repo root without any path manipulation.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="src.cython_ext.backtest_core",
    sources=["src/cython_ext/backtest_core.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="intra_backtest_core",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=False,
    ),
)
