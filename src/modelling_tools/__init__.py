# read version from installed package
from importlib.metadata import version
__version__ = version("modelling_tools")

from modelling_tools.runModel import runModel
from modelling_tools.model_summary import calculate_MSPE, calculate_CCC, plot_model_output
from modelling_tools.model import Model