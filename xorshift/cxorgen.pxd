from libc.stdint cimport uint64_t
cdef extern from "xoroshiro.h":
    ctypedef struct xor_rng_state:
        pass

    void xoroshiro_init(xor_rng_state* state, uint64_t seed);
    void xoroshiro(xor_rng_state* state, float* buf, size_t len);
    void xoroshiro_binomial(xor_rng_state* state,
                            const int N, const float p,
                            float* output, const size_t len);

    void xorshift128plus(xor_rng_state* state, float* buf, size_t len);
    void xorshift128plus_init(xor_rng_state* state, uint64_t seed);
    void xorshift128plus_binomial(xor_rng_state* state,
                                  const int N, const float p,
                                  float* output, const size_t len);
