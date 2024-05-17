"""This code can be used to create figures 5a,5b,9a,9b,9c and 9d
Change spillover_mag_k to get different values of k
First the models from the spillover...py files have to be trained to run this file
To get the graphs with the greedy algorithms, the allocations from these files also have to be run
To just get the graphs without the degree algorithms, this file can be used
"""

import os
import sys
DIR = r""
os.chdir(DIR)
sys.path.append(DIR)

import random
import numpy as np
import utils.plotting as plotting
import os
import methods.allocation.extra_allocations as extra

#Set parameters for the simulation
num_nodes = 5000 #does nothing when dataset is BC or Flickr
dataset = "full_sim" #BC, Flickr, full_sim
T = int(0.05*num_nodes)#number of treated nodes
do_greedy =  False 
do_GA = True
do_CELF = False
do_CFR = True #TARnet (alpha = 0)
do_CFR_heuristic = False  #combine degree and uplift heuristic
do_random = True
do_greedy_simulated = False
do_full = False #turn this to true if going over all budgets
get_TTE_curve_total = True #if true, TTE curve is plotted for different NT2O values
run_extra = False #if true, extra experiments are run
spillover_mag_k = 250 #k for which the effect of beta_spillover is shown


random_seed = 1000 
np.random.seed(random_seed)
random.seed(random_seed)
flipRate = 0.5
covariate_dim = 10
w_c = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of X to T
w = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of X to Y
w_beta_T2Y = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of T to Y

betaConfounding = 1 # effect of features X to T (confounding1)
betaNeighborConfounding = 0.5# effect of Neighbor features to T (confounding2)
betaTreat2Outcome = 1
bias_T2Y = 4# effect o treatment to potential outcome
#betaCovariate2Outcome =1 #effect of features to potential outcome (confounding1)

betaCovariate2Outcome =0.7
betaNeighborCovariate2Outcome =0.2 #effect of Neighbor features to potential outcome
   
    
#effect of interference
betaNeighborTreatment2Outcome = 0.3
betaNoise = 0.05 #noise
beta0 = -3#-3 #intercept
setting = dataset + "_num_nodes" + str(num_nodes) + "_T2O_" + str(betaTreat2Outcome) + "_NT2O_" + str(betaNeighborTreatment2Outcome) + "_seed_" + str(random_seed)


if dataset == "BC":
    T = 1784
    num_nodes = 1784
if dataset == "Flickr":
    T = 2358
    num_nodes = 2358

"""Allocation for node_list and spillover effect list"""



#This only works for non-greedy methods (GA,CFR,random,degree,SD)
if run_extra:
    my_node_list = [250,500,1000,2000,3000,4000]
    NT2O_list = [0,0.1,0.3,0.5,0.7]
    setting_list = []
    for NT2O in NT2O_list:
        setting = dataset + "_num_nodes" + str(num_nodes) + "_T2O_" + str(betaTreat2Outcome) + "_NT2O_" + str(NT2O) + "_seed_" + str(random_seed)
        setting_list.append(setting)
        extra.run_extra_allocations(dataset = dataset,T = T,do_GA = do_GA,do_CELF = do_CELF,do_CFR = do_CFR,do_CFR_heuristic = do_CFR_heuristic,do_greedy = do_greedy,do_random = do_random,do_greedy_simulated = do_greedy_simulated,
                                w_c = w_c,w = w,w_beta_T2Y = w_beta_T2Y,beta0 = beta0,bias_T2Y = bias_T2Y,betaTreat2Outcome = betaTreat2Outcome,betaCovariate2Outcome = betaCovariate2Outcome,
                                betaNeighborCovariate2Outcome = betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome = NT2O,betaNoise = betaNoise,setting = setting,node_list = my_node_list,do_full = do_full,num_nodes = num_nodes)


"""-------------------FIGURE GENERATION-------------------"""
if get_TTE_curve_total:
    NT2O_list = [0,0.1,0.3,0.5,0.7]
    setting_list = []
    for NT2O in NT2O_list:
        setting = dataset + "_num_nodes" + str(num_nodes) + "_T2O_" + str(betaTreat2Outcome) + "_NT2O_" + str(NT2O) + "_seed_" + str(random_seed)
        setting_list.append(setting)
    plotting.get_TTE_curve_total(k =spillover_mag_k,dataset = dataset,setting_list = setting_list,NT2O_list=NT2O_list,do_greedy=do_greedy,
                                   do_GA=do_GA,do_CELF=do_CELF,do_CFR = do_CFR,do_greedy_simulated=do_greedy_simulated)

