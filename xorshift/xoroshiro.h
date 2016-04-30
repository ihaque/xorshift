#pragma once
#include <emmintrin.h> /* SSE2 */

typedef struct _xor_rng_state {
    __m128i s0;
    __m128i s1;
} xor_rng_state;

void xoroshiro_init(xor_rng_state*, uint64_t seed);
void xoroshiro(xor_rng_state* state, float* buf, size_t len);
void xoroshiro_binomial(xor_rng_state* state,
                        const int N, const float p,
                        float* output, const size_t len);

void xorshift128plus_init(xor_rng_state* state, uint64_t seed);
void xorshift128plus(xor_rng_state* state, float* buf, size_t len);
void xorshift128plus_binomial(xor_rng_state* state,
                              const int N, const float p,
                              float* output, const size_t len);
