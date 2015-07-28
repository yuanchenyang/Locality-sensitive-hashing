from set_lsh import *
import cProfile
import random

def random_set(n=50, k=100000):
    return {random.randint(0, k) for _ in range(n)}

random_sets_100k = [random_set() for _ in range(100000)]

def bench_insert():
    l = SetLSH(5, 2)
    for s in random_sets_100k:
        l.insert(s)
    return l

def bench_query():
    l = bench_insert()
    for s in random_sets_100k:
        l.query(s)

if __name__ == '__main__':
    print 'Starting'
    cProfile.run('bench_insert()')
