# coding=utf-8

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

import os

os.system(
    'python ExecMultiview.py -log --name ModifiedMultiOmic --type .csv --views Methyl:MiRNA:RNASEQ:Clinical --pathF /home/bbauvin/Documents/Data/Data_multi_omics/ --CL_split 0.7 --CL_nbFolds 5 --CL_nb_class 2 --CL_classes Positive:Negative --CL_type Fusion --CL_cores 4 --FU_type EarlyFusion --FU_method WeightedLinear')
# /donnees/pj_bdd_bbauvin/Data_multi_omics/
#
# /home/bbauvin/Documents/Data/Data_multi_omics/
# Fusion --CL_cores 4 --FU_type EarlyFusion --FU_method WeightedLinear
# Mumbo --MU_type DecisionTree:DecisionTree:DecisionTree:DecisionTree:DecisionTree --MU_config 1:0.02 1:0.02 1:0.1 2:0.1 1:0.1 --MU_iter 1000
