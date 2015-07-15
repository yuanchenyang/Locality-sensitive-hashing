''' Locality-sensitive hashing for sets
'''

import binascii
from random import randint
from collections import defaultdict

def make_minhash(hash_family):
    '''Takes in a sequence of hash functions HASH_FAMILY, and returns a function
    that returns the minhash of its input for every element in HASH_FAMILY'''
    return lambda s: [min(hf(x) for x in s) for hf in hash_family]

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
        assert type(s) == set, 'Can only query with a set, not {}!'.format(type(s))
        minhashes = self._minhash(s)
        b, k = self.b, self.k
        digests = []

        for i in range(b):
            str_digest = ' '.join(str(minhashes[k * i + j]) for j in range(k))
            digests.append((i, self._blockhash(str_digest)))
        return digests

    def insert(self, s):
        assert type(s) == set, 'Can only insert a set, not {}!'.format(type(s))
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
            return
        best = max(candidates, key=distance_to)
        return best, distance_to(best)

if __name__ == '__main__':
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
