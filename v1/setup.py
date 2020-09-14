from distutils.core import setup, Extension

extension = Extension(
        "gslstats", 
        ["gslstats.c"],
        libraries=['gsl', 'gslcblas', 'm'],
        library_dirs=['/usr/lib', '/usr/local/lib'],
        #libraries=['libgsl.dylib', 'libgslcblas.dylib', 'libm.dylib'],
        #extra_link_args=["-L/usr/local/lib -lgsl -lgslcblas -lm"],
        #extra_compile_args=["-L/usr/local/lib -lgsl -lgslcblas -lm"],
    )

setup(
        name='gslstats', 
        version='1.0',
        ext_modules=[extension],
    )
