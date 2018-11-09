"""DGL mini-runtime"""

class Runtime(object):
    @staticmethod
    def run(execs):
        new_repr = {}
        for exe in execs:
            new_repr.update(exe.run())
        return new_repr
