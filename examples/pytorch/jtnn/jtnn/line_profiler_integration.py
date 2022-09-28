"""
line_profiler integration
"""
import os

if os.getenv("PROFILE", 0):
    import atexit

    import line_profiler

    profile = line_profiler.LineProfiler()

    profile_output = os.getenv("PROFILE_OUTPUT", None)
    if profile_output:
        from functools import partial

        atexit.register(partial(profile.dump_stats, profile_output))
    else:
        atexit.register(profile.print_stats)
else:

    def profile(f):
        return f
