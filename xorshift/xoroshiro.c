#include <assert.h>
#include <xmmintrin.h> /* SSE  */
#include <emmintrin.h> /* SSE2 */
#include "xoroshiro.h"

/* A small RNG just used to initialize the higher-quality generators
 * below.*/
static uint64_t splitmix64(uint64_t state){
	uint64_t z = (state += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}
static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}
static void xoroshiro_scalar_advance(uint64_t* s){
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];

	s1 ^= s0;
	s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s[1] = rotl(s1, 36); // c
}
void xoroshiro_init(xor_rng_state* state, uint64_t seed) {
	static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };
    uint64_t s0a, s0b, s1a, s1b;
    uint64_t gen1[2];

    /* Initialize the first generator using a splitmix as recommended */
    s0a = splitmix64(seed);
    s1a = splitmix64(s0a);

    /* Initialize the second generator (s0b, s1b) to be 2^64 next() calls
     * ahead of the first */
    gen1[0] = s0a;
    gen1[1] = s1a;
	s0b = s1b = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0b ^= gen1[0];
				s1b ^= gen1[1];
			}
			xoroshiro_scalar_advance(gen1);
		}

    state->s0 = _mm_set_epi64x(s0a, s0b);
    state->s1 = _mm_set_epi64x(s1a, s1b);
    return;
}
static void xorshift128plus_scalar_advance(uint64_t* s) {
	uint64_t s1 = s[0];
	const uint64_t s0 = s[1];
	s[0] = s0;
	s1 ^= s1 << 23; // a
	s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
}
void xorshift128plus_init(xor_rng_state* state, uint64_t seed){
	static const uint64_t JUMP[] = { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };
    uint64_t s0a, s0b, s1a, s1b;
    uint64_t gen1[2];

    /* Initialize the first generator using a splitmix as recommended */
    s0a = splitmix64(seed);
    s1a = splitmix64(s0a);

    /* Initialize the second generator (s0b, s1b) to be 2^64 next() calls
     * ahead of the first */
    gen1[0] = s0a;
    gen1[1] = s1a;
	s0b = s1b = 0;

	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0b ^= gen1[0];
				s1b ^= gen1[1];
			}
			xorshift128plus_scalar_advance(gen1);
		}

    state->s0 = _mm_set_epi64x(s0a, s0b);
    state->s1 = _mm_set_epi64x(s1a, s1b);
    return;
}

static inline __m128 random_bits_to_uniform_1_2(__m128i random_bits) {
    const __m128i SP_MASK = _mm_set1_epi32(0x3F800000);

    /* Mask off all but the mantissa bits. This is a bit wasteful of
     * the entropy, but it's quick. Then set the exponent to all 1s
     * (biased exponent) to get ^0 power.
     * This produces a uniform in the [1,2] range */

    return (__m128) _mm_or_si128(_mm_srli_epi32(random_bits, 9),
                                 SP_MASK);

}

static inline __m128 random_bits_to_uniform_0_1(__m128i random_bits) {
    return _mm_add_ps(_mm_set1_ps(-1.0f),
                      random_bits_to_uniform_1_2(random_bits));
}

static inline __m128i xoroshiro_vector_advance(__m128i* s0, __m128i* s1)
{
    __m128i xmm1 = *s0;
    __m128i xmm2 = *s1;
    __m128i xmm0, xmm3, xmm4, xmm5;

    xmm3 = xmm1;
    xmm3 = _mm_xor_si128(xmm3, xmm2);
    xmm4 = xmm3;
    xmm4 = _mm_slli_epi64(xmm4, 14);
    xmm5 = xmm1;
    xmm5 = _mm_slli_epi64(xmm5, 55);
    xmm1 = _mm_srli_epi64(xmm1, 9);
    xmm1 = _mm_or_si128(xmm1, xmm5);
    xmm1 = _mm_xor_si128(xmm1, xmm3);
    xmm1 = _mm_xor_si128(xmm1, xmm4);
    xmm2 = _mm_xor_si128(xmm2, xmm3);
    xmm2 = _mm_slli_epi64(xmm2, 36);
    xmm3 = _mm_srli_epi64(xmm3, 28);
    xmm2 = _mm_or_si128(xmm2, xmm3);
    *s0 = xmm1;
    *s1 = xmm2;
    return _mm_add_epi64(xmm1, xmm2);
}
void xoroshiro(xor_rng_state* state, float* buf, size_t len)
{
    __m128i xmm1, xmm2;
    assert(len > 0 && len % 4 == 0);

    /* Load RNG state from RAM */
    xmm1 = _mm_load_si128(&(state->s0));
    xmm2 = _mm_load_si128(&(state->s1)); 

    /* Fill buffer */
    for (;len > 0; len -= 4) {
        _mm_store_ps(buf,
                     random_bits_to_uniform_0_1(
                         xoroshiro_vector_advance(&xmm1, &xmm2)));
        buf += 4;
    }
    /* Sync stored RNG state */
    _mm_store_si128(&(state->s0), xmm1);
    _mm_store_si128(&(state->s1), xmm2);
    return;
}

static inline __m128i xorshift128plus_vector_advance(__m128i* s0,
                                                            __m128i* s1)
{
    __m128i xmm1 = *s0;
    __m128i xmm2 = *s1;
    __m128i xmm0, xmm3, xmm4, xmm5;
    xmm3 = xmm2;

    xmm4 = xmm1;
    xmm4 = _mm_slli_epi64(xmm4, 23);
    xmm1 = _mm_xor_si128(xmm1, xmm4);

    xmm5 = xmm2;
    xmm5 = _mm_srli_epi64(xmm5, 5);
    xmm2 = _mm_xor_si128(xmm2, xmm5);

    xmm4 = xmm1;
    xmm4 = _mm_srli_epi64(xmm4, 18);
    xmm1 = _mm_xor_si128(xmm1, xmm4);

    xmm2 = _mm_xor_si128(xmm2, xmm1);
    xmm1 = xmm3;
    *s0 = xmm1;
    *s1 = xmm2;
    return _mm_add_epi64(xmm1, xmm2);
}
void xorshift128plus(xor_rng_state* state, float* buf, size_t len)
{
    __m128i xmm1, xmm2;
    assert(len > 0 && len % 4 == 0);

    /* Load RNG state from RAM */
    xmm1 = _mm_load_si128(&(state->s0));
    xmm2 = _mm_load_si128(&(state->s1)); 

    /* Fill buffer */
    for (;len > 0; len -= 4) {
        _mm_store_ps(buf,
                     random_bits_to_uniform_0_1(
                         xorshift128plus_vector_advance(&xmm1, &xmm2)));
        buf += 4;
    }
    /* Sync stored RNG state */
    _mm_store_si128(&(state->s0), xmm1);
    _mm_store_si128(&(state->s1), xmm2);
    return;
}

void xoroshiro_binomial(xor_rng_state* state,
                        const int N, const float p,
                        float* output, const size_t len)
{
    const __m128 ONE = _mm_set1_ps(1.0f);
    /* Use p + 1 because we'll generate Uniform(1,2) RVs in the inner loop
     * instead of Uniform(0,1). This saves an FP add. */
    const __m128 vp = _mm_set1_ps(p+1.0f);
    __m128 accum, u_1_2, success;

    __m128i s0 = _mm_load_si128(&(state->s0));
    __m128i s1 = _mm_load_si128(&(state->s1)); 

    assert(len % 4 == 0);
    for (size_t i=0; i < len; i+=4) {
        accum = _mm_setzero_ps();
        for (int j=0; j < N; j++) {
            // Get a uniform
            u_1_2 = random_bits_to_uniform_1_2(
                        xoroshiro_vector_advance(&s0, &s1));
            // Compare to threshold and increment counter
            success = _mm_and_ps(ONE, _mm_cmplt_ps(u_1_2, vp));
            accum = _mm_add_ps(accum, success);
        }
        // accum now contains 4xBinom(N,p)
        _mm_store_ps(output + i, accum);
    }

    /* Sync stored RNG state */
    _mm_store_si128(&(state->s0), s0);
    _mm_store_si128(&(state->s1), s1);
}
void xorshift128plus_binomial(xor_rng_state* state,
                        const int N, const float p,
                        float* output, const size_t len)
{
    const __m128 ONE = _mm_set1_ps(1.0f);
    /* Use p + 1 because we'll generate Uniform(1,2) RVs in the inner loop
     * instead of Uniform(0,1). This saves an FP add. */
    const __m128 vp = _mm_set1_ps(p + 1.0f);
    __m128 accum, u_1_2, success;

    __m128i s0 = _mm_load_si128(&(state->s0));
    __m128i s1 = _mm_load_si128(&(state->s1)); 

    assert(len % 4 == 0);
    for (size_t i=0; i < len; i+=4) {
        accum = _mm_setzero_ps();
        for (int j=0; j < N; j++) {
            // Get a uniform
            u_1_2 = random_bits_to_uniform_1_2(
                        xorshift128plus_vector_advance(&s0, &s1));
            // Compare to threshold and increment counter
            success = _mm_and_ps(ONE, _mm_cmplt_ps(u_1_2, vp));
            accum = _mm_add_ps(accum, success);
        }
        // accum now contains 4xBinom(N,p)
        _mm_store_ps(output + i, accum);
    }

    /* Sync stored RNG state */
    _mm_store_si128(&(state->s0), s0);
    _mm_store_si128(&(state->s1), s1);
}
