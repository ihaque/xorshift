cimport cxorgen
from os import urandom
import numpy as np
cimport numpy as np

def get_seed(seed=None):
    if seed is None:
        rand_bytes = urandom(8)
        seed = 0
        for i in xrange(8):
            seed = seed << 8
            seed = seed | ord(rand_bytes[i])
    else:
        seed = hash(seed)
    return seed


cdef class Xoroshiro:
    cdef cxorgen.xor_rng_state state
    cdef np.ndarray buf
    def __cinit__(self, seed=None):
        cxorgen.xoroshiro_init(&self.state, get_seed(seed))

    def __init__(self, seed=None):
        self.buf = np.empty((0,), dtype=np.float32)
        pass

    cpdef size_t reallocate(self, size_t num):
        cdef ssize_t ceil_num = (num / 4) * 4
        if num % 4:
            ceil_num += 4
        if self.buf.shape[0] < ceil_num:
            self.buf = np.empty((ceil_num,), dtype=np.float32)
        return ceil_num

    cpdef np.ndarray uniform(self, size_t num):
        cdef size_t ceil_num = self.reallocate(num)

        cxorgen.xoroshiro(&self.state, <float*> self.buf.data, ceil_num)

        return self.buf[:num]

    cpdef np.ndarray binomial(self,size_t N, float p, size_t num):
        cdef size_t ceil_num = self.reallocate(num)

        cxorgen.xoroshiro_binomial(&self.state, N, p,
                                   <float*> self.buf.data,
                                   ceil_num)
        return self.buf[:num]

cdef class Xorshift128plus:
    cdef cxorgen.xor_rng_state state
    cdef np.ndarray buf
    def __cinit__(self, seed=None):
        cxorgen.xorshift128plus_init(&self.state, get_seed(seed))

    def __init__(self, seed=None):
        self.buf = np.empty((0,), dtype=np.float32)
        pass

    cpdef size_t reallocate(self, size_t num):
        cdef ssize_t ceil_num = (num / 4) * 4
        if num % 4:
            ceil_num += 4
        if self.buf.shape[0] < ceil_num:
            self.buf = np.empty((ceil_num,), dtype=np.float32)
        return ceil_num

    cpdef np.ndarray uniform(self, size_t num):
        cdef size_t ceil_num = self.reallocate(num)
        cxorgen.xorshift128plus(&self.state, <float*> self.buf.data, ceil_num)

        return self.buf[:num]

    cpdef np.ndarray binomial(self,size_t N, float p, size_t num):
        cdef size_t ceil_num = self.reallocate(num)

        cxorgen.xorshift128plus_binomial(&self.state, N, p,
                                   <float*> self.buf.data,
                                   ceil_num)
        return self.buf[:num]

