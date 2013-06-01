__all__ = ['TestCase']

import unittest
from fickle import backend


class TestCase(unittest.TestCase):

    def setUp(self):
        backend.CACHE.delete(backend.CACHE_KEY)
