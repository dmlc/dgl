"""DGL mini-runtime."""

class Runtime(object):
    @staticmethod
    def run(prog):
        for exe in prog.execs:
            exe.run()
