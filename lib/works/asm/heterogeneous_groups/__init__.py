"""
Heterogeneous groups
"""

from importlib.metadata import version

from . import data_properties, grouping

try:
    __version__ = version(__name__)
except: # pylint: disable=bare-except
    __version__ = "0.0.0"
