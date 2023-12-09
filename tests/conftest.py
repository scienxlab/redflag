import doctest
import re
import platform


OutputChecker = doctest.OutputChecker
class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        """
        Remove the dtype from NumPy array reprs, to avoid some doctests
        failing on Windows, which often uses int32 instead of int64.
        """
        pattern = re.compile(r"(array\(.+?)(, dtype=int)(32|64)(\))")
        want = pattern.sub(r"\1\4", want)
        got = pattern.sub(r"\1\4", got)
        return OutputChecker.check_output(self, want, got, optionflags)

if platform.system() == 'Windows':
    doctest.OutputChecker = CustomOutputChecker
