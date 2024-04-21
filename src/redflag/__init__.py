from .utils import *
from .sklearn import *
from .pandas import *
from .markov import Markov_chain

# Targets
from .target import *
from .imbalance import *

# Features
from .distributions import *
from .independence import *
from .importance import *
from .outliers import *

# It used to be conventional to define a __version__ attribute.
# However, it is now considered best practice to get version
# information from the package metadata directly, eg by using
# importlib.metadata.version (see below).
#
# This will be deprecated in v0.5.0 but for now we do this:
#
from importlib.metadata import version
__version__ = version(__package__ or __name__)
