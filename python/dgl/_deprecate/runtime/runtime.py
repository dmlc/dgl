"""DGL mini-runtime."""


class Runtime(object):
    """The mini runtime class."""
    @staticmethod
    def run(prog):
        """Run the given program."""
        for exe in prog.execs:
            # prog.pprint_exe(exe)
            exe.run()
