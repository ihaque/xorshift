from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

# Suppress unused function warnings from clang arising in numpy code
os.environ['CFLAGS'] = '%s %s' % (os.environ.get('CFLAGS', ''), '-Wno-unused-function')

setup(
    name='xorshift',
    version='0.1.0',
    description='Vectorized xorshift and xoroshiro uniform/binomial RNGs',
    author='Imran Haque',
    author_email='ihaque@cs.stanford.edu',
    license='MIT',
    ext_modules = cythonize(
        [Extension("xorshift.xorgen",
                   ["xorshift/xorgen.pyx", "xorshift/xoroshiro.c"],
                   include_dirs=[numpy.get_include()])]),
    packages=['xorshift'],
)
