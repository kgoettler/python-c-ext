#!/usr/bin/env python3

from distutils.core import setup, Extension
import numpy as np

extension = Extension(
        "gslstats", 
        ["gslstats.c", "ttest.c"],
        libraries=['gsl', 'gslcblas', 'm'],
        library_dirs=['/usr/lib', '/usr/local/lib'],
        include_dirs=[np.get_include()],
    )

setup(
        name='gslstats', 
        version='1.0',
        ext_modules=[extension],
    )
