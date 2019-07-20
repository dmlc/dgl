"""Environment & config functions and variables"""
import os

def get_backend():
    """Return the current backend."""
    return os.getenv('DGLBACKEND', 'pytorch')
