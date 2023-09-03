from .utils import *
from .sklearn import *

# Targets
from .target import *
from .imbalance import *

# Features
from .distributions import *
from .independence import *
from .importance import *
from .outliers import *

# From https://github.com/pypa/setuptools_scm
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    pass
