__all__ = ['Redis', 'DATASET_KEY', 'MODEL_KEY', 'read_key', 'write_key']

import os
import pickle
import zlib

import redis

DATASET_KEY = 'fickle:dataset'
MODEL_KEY = 'fickle:model'

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
Redis = redis.from_url(REDIS_URL)


def read_key(key):
    try:
        string = Redis.get(key)
    except redis.ConnectionError:
        return
    if not string:
        return
    obj = pickle.loads(zlib.decompress(string))
    return obj


def write_key(key, obj):
    string = zlib.compress(pickle.dumps(obj))
    Redis.set(key, string)
