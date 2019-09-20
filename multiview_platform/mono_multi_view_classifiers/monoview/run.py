# coding=utf-8
import os

os.system(
    'python exec_classif_mono_view.py -log --name MultiOmicDataset --type hdf5 --feat RNASeq --pathF /home/doob/Téléchargements/Data_multi_omics/ --CL_type DecisionTree --CL_CV 5 --CL_Cores 4 --CL_split 0.5')
# /donnees/pj_bdd_bbauvin/Data_multi_omics/
# MiRNA_  RNASeq  Clinic
#
