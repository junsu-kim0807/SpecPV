import time
from collections import defaultdict
from functools import wraps

import torch


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f"{self.name} took {elapsed} seconds")


_time_records = defaultdict(list)


def record_time(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            _time_records[name].append(elapsed)
            return result

        return wrapper

    return decorator


def print_time_stats():
    for name, times in _time_records.items():
        avg = sum(times) / len(times)
        print(
            f"{name}: called {len(times)} times, avg {avg:.6f}s, min {min(times):.6f}s, max {max(times):.6f}s"
        )


def reset_time_stats():
    _time_records.clear()
