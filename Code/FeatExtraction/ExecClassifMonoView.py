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

parser = ArgumentParser(description='Perform feature parameter optimisation')

parser.add_argument('-p', '--path', action='store', help='Path to the database', default='D:\\CaltechMini')
parser.add_argument('-c', '--cores', action='store', type=int, help='Nb cores used for parallelization', default=1)

args = parser.parse_args()