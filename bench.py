import numpy as np
from time import time
import xorshift

rng = xorshift.Xoroshiro()
rng2 = xorshift.Xorshift128plus()

def output(name, start, end):
    elapsed = (end - start) * 1000
    per_iter = elapsed / iters
    per_rv = per_iter / count * 1e6
    print '%s took %.2f ms/iter, %.2f ns per float' % (name, per_iter, per_rv)

def bench_binomial(iters, count, N, p):
    print "Benchmarking generation of %d Bin(%d,%f) RVs, %d iterations" % (
            count, N, p, iters)
    print "------------------------------"

    start = time()
    for i in xrange(iters):
        np.random.binomial(N, p, count)
    end = time()
    output('numpy', start, end)

    start = time()
    for i in xrange(iters):
        rng.binomial(N, p, count)
    end = time()
    output('xoroshiro', start, end)

    start = time()
    for i in xrange(iters):
        rng2.binomial(N, p, count)
    end = time()
    output('xoroshift128plus', start, end)

def bench_uniform(iters, count):
    print "Benchmarking generation of %d Uniform(0,1) RVs, %d iterations" % (
            count, iters)
    print "------------------------------"

    start = time()
    for i in xrange(iters):
        np.random.uniform(size=count)
    end = time()
    output('numpy', start, end)

    start = time()
    for i in xrange(iters):
        rng.uniform(size=count)
    end = time()
    output('xoroshiro', start, end)

    start = time()
    for i in xrange(iters):
        rng2.uniform(size=count)
    end = time()
    output('xoroshift128plus', start, end)

iters = 10
count = 131072
N = 50
p = 0.25

bench_binomial(iters, count, N, p)
print
bench_uniform(iters, count)
