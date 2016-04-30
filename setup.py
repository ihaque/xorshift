from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    name='xorshift',
    version='0.1.0',
    description='Vectorized xorshift and xoroshiro uniform/binomial RNGs',
    author='Imran Haque',
    author_email='ihaque@cs.stanford.edu',
    license='MIT',
    ext_modules = cythonize([Extension("xorgen", ["xorgen.pyx", "xoroshiro.c"],
                             #extra_objects=["xoroshiro.o"],
                             include_dirs=[numpy.get_include()])])
)
