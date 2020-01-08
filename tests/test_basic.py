import unittest
from mpimap import Pool

class BasicTest(unittest.TestCase):
    def test_pool(self):
        def fct(idx):
            return idx
        with Pool() as p:
            results = p.map(fct, range(1000))
        self.assertEqual(results, list(range(1000)))
