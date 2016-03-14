#!/usr/bin/env python

""" Script to perform feature parameter optimisation """

# Import built-in modules
from argparse import ArgumentParser # for acommand line arguments

# Import 3rd party modules

# Import own modules


# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-03-10

### Argument Parser
parser = argparse.ArgumentParser(
description='This methods permits to export one or more features at the same time for a database of images (path, name). To extract one feature activate it by using the specific argument (e.g. -RGB). For each feature you can define the parameters by using the optional arguments (e.g. --RGB_Hist 32). The results will be exported to a CSV-File.', 
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('necessary arguments:')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Select a name, e.g. Caltech (default: %(default)s)', default='DB')
groupStandard.add_argument('--path', metavar='STRING', action='store', help='Path to the database (default: %(default)s)', default='D:\\CaltechMini')
groupStandard.add_argument('--cores', metavar='INT', action='store', help='Number of cores for HOG', type=int, default=1)


groupRGB = parser.add_argument_group('RGB arguments:')
groupRGB.add_argument('-RGB', action='store_true', help='Use option to activate RGB')
groupRGB.add_argument('--RGB_Hist', metavar='INT', action='store', help='Number of bins for histogram', type=int, default=16)
groupRGB.add_argument('--RGB_CI', metavar='INT', action='store', help='Max Color Intensity [0 to VALUE]', type=int, default=256)
groupRGB.add_argument('-RGB_NMinMax', action='store_true', help='Use option to actvate MinMax Norm instead of Distribution')