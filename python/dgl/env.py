import os

def get_backend():
    return os.getenv('DGLBACKEND', 'pytorch')
