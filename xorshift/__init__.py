from operator import mul

import xorshift.xorgen


def _compute_n_elements(shape):
    if shape is None:
        shape = (1,)
    try:
        nelts = reduce(mul, shape)
    except TypeError:
        nelts = int(shape)
        shape = (shape, )
    return nelts, shape


class _Generator(object):
    def uniform(self, low=0.0, high=1.0, size=None):
        nelts, size = _compute_n_elements(size)

        rvs = self.rng.uniform(nelts)

        if (high - low) != 1.0:
            rvs *= (high - low)
        if low != 0.0:
            rvs += low

        return rvs.reshape(size)

    def binomial(self, N, p, size=None):
        nelts, size = _compute_n_elements(size)
        
        return self.rng.binomial(N, p, nelts).reshape(size)


class Xoroshiro(_Generator):
    def __init__(self, seed=None):
        self.rng = xorgen.Xoroshiro(seed)


class Xorshift128plus(_Generator):
    def __init__(self, seed=None):
        self.rng = xorgen.Xorshift128plus(seed)


__all__ = [Xoroshiro, Xorshift128plus]
