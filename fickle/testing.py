__all__ = ['TestCase']

import unittest

from .storage import *


class TestCase(unittest.TestCase):

    def setUp(self):
        Redis.delete(MODEL_KEY)
        Redis.delete(DATASET_KEY)
