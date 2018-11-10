"""DGL mini-runtime"""

class Runtime(object):
    @staticmethod
    def run(execs):
        for exe in execs:
            exe.run()
