import os
os.system('python ExecMultiview.py -log --name MultiOmic --type .hdf5 --views Methyl:MiRNA:RNASEQ:Clinical --pathF /home/bbauvin/Documents/Data/Data_multi_omics/ --CL_split 0.3 --CL_nbFolds 4 --CL_nb_class 2 --CL_classes Positive:Negative --CL_type Fusion --CL_cores 4 --FU_type EarlyFusion --FU_method WeightedLinear')
# /donnees/pj_bdd_bbauvin/Data_multi_omics/
# --MU_type DecisionTree:DecisionTree:DecisionTree:DecisionTree --MU_config 1:0.015 1:0.015 1:0.1 2:0.3 --MU_iter 100