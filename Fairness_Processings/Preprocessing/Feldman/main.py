from Feldman import Feldman
import helper as h
dataset = 'adult' #compas, german
rep = 1.0 #repair level
f_name = None

Feldman(dataset, rep, f_name)
h.Adult("results_Feldman/adult_train_repaired.csv")
