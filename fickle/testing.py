__all__ = ['TestCase']

import unittest

from cache import *


class TestCase(unittest.TestCase):

    def setUp(self):
        CACHE.delete(CACHE_KEY)
