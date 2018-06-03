"""
MBAR_PyTorch is an implementation of the multistate Bennette acceprance ratio 
(MBAR) [1] method using the PyTorch [2] library. Comparing with the package 
pymbar [3], MBAR_PyTorch is faster when calculating free energyies for a large
 num of states with a large num of conformations.
"""
from MBAR_PyTorch.MBAR import MBAR
from MBAR_PyTorch.MBAR import test
