from set_lsh import *
import cProfile
import random

random_sets = [{random.randint(0, 100000) for _ in range(50)} for _ in range(100000)]

def bench_insert():
    l = SetLSH(5, 2)
    for s in random_sets:
        l.insert(s)

if __name__ == '__main__':
    cProfile.run('bench_insert()')
