# xorshift: Fast vectorized RNGs for Python

C/Python implementation of Sebastiano Vigna's [xoroshiro128+ and xorshift128+](http://xorshift.di.unimi.it/) random number generators. These are fast, high quality random bit generators with shorter periods (2^128 - 1) than the usual Mersenne Twister, but also fewer failures on [TestU01](http://www.iro.umontreal.ca/~simardr/testu01/tu01.html).

This implementation vectorizes the 64-bit implementations of the xoroshiro128+
and xorshift128+ generators to produce 4 single precision floats per iteration.
They're *fast*: about 5 instructions per uniform float in the inner loop:

(Benchmark results from a 3.1GHz Core i7 13" Retina MacBook Pro, Early 2015)
```
Benchmarking generation of 131072 Bin(50, 0.25) RVs, 10 iterations
------------------------------------------------------------------
numpy took 15.90 ms/iter,           121.30 ns per float
xoroshiro took 3.88 ms/iter,        29.61 ns per float
xoroshift128plus took 3.46 ms/iter, 26.41 ns per float

Benchmarking generation of 131072 Uniform(0,1) RVs, 10 iterations
-----------------------------------------------------------------
numpy took 1.71 ms/iter,            13.07 ns per float
xoroshiro took 0.11 ms/iter,        0.82 ns per float
xoroshift128plus took 0.10 ms/iter, 0.76 ns per float
```

Currently, a subset of the `numpy.random.RandomState` interface is implemented:
`Xoroshiro` or `Xorshift128plus` generators can be created with a seed (or no
seed, seeding from `os.urandom()`, and they can generate single-precision
uniform or binomial variates, of arbitrary shape.

Additionally, the generators
can be used in a "no-copy" mode, in which the returned random variates are a
window into an internal buffer. This can be useful if you will immediately use
the numbers or will directly copy them into another buffer, to avoid an
intermediate memory allocation and copy.

## API

`xorshift.Xoroshiro` and `xorshift.Xorshift128plus` implement the following
`Generator` interface:

### Methods
#### `__init__(self, seed=None, copy=True)`:

Initialize the generator with the given `seed`. `seed` may be `None` or any
object defining the `__hash__` magic method. If `None`, the generator will seed
itself from `os.urandom()`.

If `copy` is `False`, the returned random numbers will be a window into an
internally allocated memory buffer; if `copy` is `False`, the returned random
numbers will be a freshly allocated ndarray. `copy=False` is faster, but can
lead to errors:

```python
rng_nocopy = xorgen.Xoroshiro(copy=False)

nums_1 = rng_nocopy.uniform(size=4)
nums_2 = rng_nocopy.uniform(size=4)
# nums_1 and nums_2 are now windows into the same array, not independent arrays
```

`copy=False` can be useful if you will immediately use the results of the RNG or
if you will copy them into your own destination:

```python
scatter_array[indices] = rng_nocopy(size=len(indices))
```

#### `uniform(low=0.0, high=1.0, size=None)`
See [numpy.random.uniform](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.uniform.html)

#### `binomial(N, p, size=None)`
See [numpy.random.binomial](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.binomial.html)
