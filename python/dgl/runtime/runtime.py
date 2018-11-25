"""DGL mini-runtime."""

class Runtime(object):
    @staticmethod
    def run(prog):
        for exe in prog.execs:
            #prog.pprint_exe(exe)
            exe.run()
