"""DGL mini-runtime."""

class Runtime(object):
    @staticmethod
    def run(prog):
        for inst in prog.insts:
            inst.run()
