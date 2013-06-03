__all__ = ['CACHE', 'CACHE_KEY']

import os

import redis

CACHE_KEY = 'fickle:model'
CACHE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
CACHE = redis.from_url(CACHE_URL)
