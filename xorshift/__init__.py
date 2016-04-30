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

        rvs = rvs.reshape(size)
        if self.copy:
            return np.copy(rvs)
        else:
            return rvs

    def binomial(self, N, p, size=None, copy=False):
        nelts, size = _compute_n_elements(size)
        
        rvs = self.rng.binomial(N, p, nelts).reshape(size)
        if self.copy:
            return np.copy(rvs)
        else:
            return rvs


class Xoroshiro(_Generator):
    def __init__(self, seed=None, copy=True):
        self.copy = copy
        self.rng = xorgen.Xoroshiro(seed)


class Xorshift128plus(_Generator):
    def __init__(self, seed=None, copy=True):
        self.copy = copy
        self.rng = xorgen.Xorshift128plus(seed)


__all__ = [Xoroshiro, Xorshift128plus]
