import unittest

import numpy as np

from xorshift import Xoroshiro
from xorshift import Xorshift128plus


class TestSeeding(unittest.TestCase):
    def test_repeatable_seeding(self):
        def assert_repeatable_seeding(cls):
            SEED = 1
            rng1 = cls(seed=SEED)
            rng2 = cls(seed=SEED)
            self.assertEqual(rng1.uniform(), rng2.uniform())
            self.assertEqual(rng1.binomial(1000, 0.5), rng2.binomial(1000, 0.5))
        assert_repeatable_seeding(Xoroshiro)
        assert_repeatable_seeding(Xorshift128plus)

    def test_distinct_seeds(self):
        def assert_distinct_seeds(cls):
            rng1 = cls(seed=1)
            rng2 = cls(seed=2)
            self.assertNotEqual(rng1.uniform(),
                                rng2.uniform())
            self.assertNotEqual(rng1.binomial(1000, 0.5),
                                rng2.binomial(1000, 0.5))
        assert_distinct_seeds(Xoroshiro)
        assert_distinct_seeds(Xorshift128plus)


class TestUniform(unittest.TestCase):
    def test_uniform_range(self):
        def assert_range(cls):
            seed = 1
            size = 100000
            rng = cls(seed=1)
            uniforms = rng.uniform(size=size)
            self.assertGreaterEqual(uniforms.min(), 0)
            self.assertLessEqual(uniforms.max(), 1)

            uniforms = rng.uniform(high=2.0, size=size)
            self.assertGreaterEqual(uniforms.min(), 0)
            self.assertLessEqual(uniforms.max(), 2)
            self.assertGreaterEqual(uniforms.max(), 1)

            uniforms = rng.uniform(low=-1.0,high=2.0, size=size)
            self.assertGreaterEqual(uniforms.min(), -1.0)
            self.assertLessEqual(uniforms.max(), 2)
            self.assertLessEqual(uniforms.min(), 0)
            self.assertGreaterEqual(uniforms.max(), 1)
        assert_range(Xoroshiro)
        assert_range(Xorshift128plus)
    def test_uniform_moments(self):
        def assert_mean_variance(cls):
            seed = 1
            size = 1000000
            low = 0.0
            high = 1.0
            expected_mean = 0.5 * (low + high)
            expected_variance = (1./12) * ((high - low) ** 2)

            rng = cls(seed=1)
            rvs = rng.uniform(low, high, size=size)
            mean = rvs.mean()
            variance = rvs.std() ** 2

            abs_diff_mean = abs(mean - expected_mean)
            abs_diff_var = abs(variance - expected_variance)

            # Mean to within .1%
            self.assertLessEqual(abs_diff_mean, .001 * expected_mean)
            # Variance to within 1%
            self.assertLessEqual(abs_diff_var, .01 * expected_mean)
        assert_mean_variance(Xoroshiro)
        assert_mean_variance(Xorshift128plus)


class TestBinomial(unittest.TestCase):
    def test_binomial_moments(self):
        def assert_mean_variance(cls):
            seed = 1
            size = 1000000
            N = 10
            p = 0.5
            expected_mean = N*p
            expected_variance = N*p*(1-p)

            rng = cls(seed=1)
            binoms = rng.binomial(N, p, size=size)
            mean = binoms.mean()
            variance = binoms.std() ** 2

            abs_diff_mean = abs(mean - expected_mean)
            abs_diff_var = abs(variance - expected_variance)

            # Mean to within .1%
            self.assertLessEqual(abs_diff_mean, .001 * expected_mean)
            # Variance to within 1%
            self.assertLessEqual(abs_diff_var, .01 * expected_mean)
        assert_mean_variance(Xoroshiro)
        assert_mean_variance(Xorshift128plus)

class TestCopy(unittest.TestCase):
    def test_nocopy(self):
        def assert_view_uniform(cls):
            seed = 1
            rng = cls(seed, copy=False)
            rvs1 = rng.uniform(size=4)
            rvs2 = rng.uniform(size=4)
            self.assertTrue(np.all(rvs1 == rvs2))
        def assert_view_binomial(cls):
            seed = 1
            rng = cls(seed, copy=False)
            rvs1 = rng.binomial(1000, 0.5, size=4)
            rvs2 = rng.binomial(1000, 0.5, size=4)
            self.assertTrue(np.all(rvs1 == rvs2))
        def assert_copy_uniform(cls):
            seed = 1
            rng = cls(seed, copy=True)
            rvs1 = rng.uniform(size=4)
            rvs2 = rng.uniform(size=4)
            self.assertFalse(np.all(rvs1 == rvs2))
        def assert_copy_binomial(cls):
            seed = 1
            rng = cls(seed, copy=True)
            rvs1 = rng.binomial(1000, 0.5, size=4)
            rvs2 = rng.binomial(1000, 0.5, size=4)
            self.assertFalse(np.all(rvs1 == rvs2))

        assert_view_uniform(Xoroshiro)
        assert_copy_uniform(Xoroshiro)
        assert_view_binomial(Xoroshiro)
        assert_copy_binomial(Xoroshiro)
        assert_view_uniform(Xorshift128plus)
        assert_copy_uniform(Xorshift128plus)
        assert_view_binomial(Xorshift128plus)
        assert_copy_binomial(Xorshift128plus)
