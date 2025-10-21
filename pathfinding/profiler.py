import time
pathfinding_profiling_enable = True

def profile(func):
    def wrapper(*args, **kwargs):
        if not pathfinding_profiling_enable:
            return func(*args, **kwargs)

        t = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t) * 1000  # en ms
        print(f"[PROFILE] {func.__name__:<25}: {elapsed:7.2f} ms")
        return result
    return wrapper