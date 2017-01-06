''' Locality-sensitive hashing for sets
'''

import binascii
from random import randint
from collections import defaultdict

import numpy as np


class WeightedSet(dict):
    def normalize(self):
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

def weighted_jaccard(A, B):
    assert type(A) == WeightedSet and type(B) == WeightedSet,\
        'Inputs must be WeightedSets!'
    return sum([min(A[key], B[key]) for key in A.keys()]) / \
           sum([max(A[key], B[key]) for key in A.keys()])


def make_weightedminhash(hash_family):
    """
    Takes in a sequence of hash functions hash_family, and returns a function that
    returns the weightedminhash of its input for every element in hash_family.

    Algorithm and notation detailed in:
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5693978
    :return:
    """
    def h(hf, s, tol=1e-10):
        # FIXME use of s.values() is awkward. Re-implement as list instead of dict?
        # Use (np.inf, np.inf) as default large values where x <= 0
        hfs = zip(*[hf(x) if np.abs(x) > tol else (np.inf, np.inf) for x in s.values()])
        argmin_a = np.argmin(hfs[1])
        min_t = hfs[0][argmin_a]
        return argmin_a, min_t

    return lambda s: [h(hf, s) for hf in hash_family]


def weightedminhash_gen():
    """
    Returns a random consistent hash function for the weighted min hash.

    Algorithm and notation detailed in:
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5693978
    :return:
    """
    # if k != 1:
    #     raise NotImplementedError, "k > 1 digit hashes, not yet implemented"

    r = np.random.gamma(2, 1)
    c = np.random.gamma(2, 1)
    beta = np.random.uniform(0, 1)

    def h(x):
        t = np.floor(np.log(x)/r + beta)
        a = c / (np.exp(r*(t-beta) + r))
        return t,a

    return h


def make_minhash(hash_family):
    '''Takes in a sequence of hash functions HASH_FAMILY, and returns a function
    that returns the minhash of its input for every element in HASH_FAMILY'''
    return lambda s: [min([hf(x) for x in s]) for hf in hash_family]

def hash_family(hash_gen, n):
    '''Creates a random family of hashes'''
    return [hash_gen() for _ in range(n)]

def default_gen():
    '''Returns a random hash function based on python's hash in the range
    [0, sys.maxint], 3 times faster than md5/sha1 '''
    prime = 2**31 - 1
    mask1, mask2 = randint(0, prime), randint(0, prime)
    return lambda x: ((hash(x) + mask1) * mask2) % prime

def jaccard(A, B):
    assert type(A) == set and type(B) == set, 'Inputs must be sets!'
    return len(A.intersection(B)) / float(len(A.union(B)))


class SetLSH:
    def __init__(self, b, k, hashgen=None, blockhash=None):
        self._hash_type = set

        # Constants for LSH, see ipython notebook
        # We have b blocks of k hashes each
        self.b = b
        self.k = k

        # Increments with each insertion
        self._item_id = 0

        # Default to default_gen for minhashing, can replace with custom hash
        # function suited to data. Collisions will affect accuracy.
        self._minhash = make_minhash(hash_family(hashgen or default_gen, b * k))

        # Default to crc32 for hashing blocks, we want to avoid collisions here
        # as much as possible yet still be fast, to improve lookup times.
        self._blockhash = blockhash or binascii.crc32

        # Dictionary where we collect candidates for Jaccard similarity
        self._rowhashes = {}

        # We store the actual inserted sets here
        self._sets = []

    def _get_digests(self, s):
        assert type(s) == self._hash_type, 'Can only query with a set, not {}!'.format(type(s))
        minhashes = self._minhash(s)
        b, k = self.b, self.k
        digests = []

        for i in range(b):
            str_digest = ' '.join(str(minhashes[k * i + j]) for j in range(k))
            digests.append((i, self._blockhash(str_digest)))
        return digests

    def insert(self, s):
        assert type(s) == self._hash_type, 'Can only insert a set, not {}!'.format(type(s))
        self._sets.append(s)
        for digest in self._get_digests(s):
            self._rowhashes.setdefault(digest, []).append(self._item_id)
        self._item_id += 1

    def get_candidates(self, s, index=False):
        candidates = set()
        for digest in self._get_digests(s):
            for set_index in self._rowhashes.get(digest, []):
                candidates.add(set_index)
        return [c if index else self._sets[c] for c in candidates]

    def query(self, s, metric=None, index=False):
        distance_to = lambda x: (metric or jaccard)(self._sets[x]
                                                    if index else x, s)
        candidates = self.get_candidates(s, index=index)
        if candidates == []:
            return set(), 0
        best = max(candidates, key=distance_to)
        return best, distance_to(best)


class WeightedSetLSH(SetLSH):
    def __init__(self, b, k, hashgen=None, blockhash=None):
        SetLSH.__init__(self, b, k, hashgen=hashgen, blockhash=blockhash)
        self._hash_type = WeightedSet

        self._minhash = make_weightedminhash(hash_family(hashgen or
                                             weightedminhash_gen, b))

    def _get_digests(self, s):
        assert type(s) == self._hash_type, 'Can only query with a WeightedSet, not {}!'.format(type(s))
        minhashes = self._minhash(s)
        b, k = self.b, self.k
        digests = []

        for i in range(b):
            str_digest = str(minhashes[i])
            digests.append((i, self._blockhash(str_digest)))
        return digests


if __name__ == '__main__':
    print "Set LSH example"
    s = SetLSH(5, 2)
    s.insert({1, 2, 3, 4})
    s.insert({4, 5, 6, 7})
    s.insert({6, 7, 8, 9})
    s.insert({2, 4, 6, 8})
    print s._rowhashes
    print s._rowhashes.values()
    print s.get_candidates({2, 4, 5})
    print s.query({2, 4, 5})
    print s.query({2, 4, 5}, index=True)

    print
    print "Weighted Set LSH example"
    s = WeightedSetLSH(5, 1)
    s.insert(WeightedSet({0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}))
    s.insert(WeightedSet({0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0}))
    s.insert(WeightedSet({0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0}))
    s.insert(WeightedSet({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}))
    s.insert(WeightedSet({0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}))
    s.insert(WeightedSet({0: 0.1, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2}))
    s.insert(WeightedSet({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.0}))
    s.insert(WeightedSet({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.19999, 4: 0.00001}))
    print s._rowhashes
    print s._rowhashes.values()
    query = WeightedSet({0: 0.0, 1: 0.8, 2: 0.0, 3: 0.0, 4: 0.2})
    print s.get_candidates(query)
    print s.query(query, metric=weighted_jaccard)
