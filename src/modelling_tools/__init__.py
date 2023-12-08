# read version from installed package
from importlib.metadata import version
__version__ = version("modelling_tools")

from modelling_tools.runModel import runModel