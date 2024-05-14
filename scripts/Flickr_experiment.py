"""By running this code, Figure 4a, 4c, and Table 5c from the paper can be reproduced. """

import os
import sys
DIR = r""
os.chdir(DIR)
sys.path.append(DIR)

import random
import numpy as np
import data.data_generator as data_generator
import methods.causal_models.model_tuning as model_tuning
import methods.allocation.get_allocations as run_allocations
import utils.plotting as plotting
import os
import yaml

#Set parameters for the simulation
num_nodes = 2358 #does nothing when dataset is BC or Flickr
dataset = "Flickr" #BC, Flickr, full_sim
T = int(0.05*num_nodes) #number of treated nodes
do_greedy =  True 
do_GA = True
do_CELF = True
do_CFR = True #TARnet (alpha = 0)
do_CFR_heuristic = False  #combine degree and uplift heuristic (leave on False)
do_random = True
do_greedy_simulated = True
do_full = True #turn this to true if going over all budgets
do_only_allocations = False #if true, the data generation and training step are skipped
do_only_graphs = False #if true allocations are skipped
get_degree_distribution = True #if true, degree distribution is plotted
similarity_k = 118 #k for which the similarity matrix is generated (118,472,943)


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


"""-------------------DATA GENERATION-------------------"""
if not do_only_allocations:
    

    data_generator.simulate_data(dataset= dataset,w_c = w_c,w = w,w_beta_T2Y = w_beta_T2Y,betaConfounding = betaConfounding,betaNeighborConfounding = betaNeighborConfounding,
                        betaTreat2Outcome = betaTreat2Outcome,bias_T2Y = bias_T2Y,betaCovariate2Outcome = betaCovariate2Outcome,betaNeighborCovariate2Outcome = betaNeighborCovariate2Outcome,
                        betaNeighborTreatment2Outcome = betaNeighborTreatment2Outcome,betaNoise = betaNoise,beta0 = beta0,nodes = num_nodes,flipRate = flipRate,covariate_dim = covariate_dim,setting = setting)

    #Hyperparameter tuning for each dataset
    """-------------------MODEL TRAINING-------------------"""
    config_netest, val_loss_netest, config_CFR, val_loss_CFR =  model_tuning.train_best_models(dataset = dataset,setting = setting)
    directory = "models/" + setting + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    print ("Best config for NetEstimator: ",config_netest)
    file_path = directory + "best_config_netest.yaml"
    with open(file_path, 'w') as file:
        yaml.dump(config_netest, file)
    
    print ("Best config for CFR: ",config_CFR)
    
    file_path = directory + "best_config_CFR.yaml"
    with open(file_path, 'w') as file:
        yaml.dump(config_CFR, file)


"""-------------------ALLOCATIONS-------------------"""
# Specify the directory path
if dataset == "BC":
    T = 1784
    num_nodes = 1784
if dataset == "Flickr":
    T = 2358
    num_nodes = 2358
if not do_only_graphs:
    run_allocations.run_allocations(dataset=dataset,T=T,do_greedy=do_greedy,do_GA=do_GA,do_CELF=do_CELF,do_CFR = do_CFR,do_CFR_heuristic=do_CFR_heuristic,
                                do_random =do_random, do_greedy_simulated=do_greedy_simulated,w_c = w_c,w = w,w_beta_T2Y = w_beta_T2Y,
                                betaTreat2Outcome = betaTreat2Outcome,bias_T2Y = bias_T2Y,betaCovariate2Outcome = betaCovariate2Outcome,betaNeighborCovariate2Outcome = betaNeighborCovariate2Outcome,
                                betaNeighborTreatment2Outcome = betaNeighborTreatment2Outcome,betaNoise = betaNoise,beta0 = beta0,setting = setting,do_full= do_full,
                                num_nodes=num_nodes) 

"""-------------------FIGURE GENERATION-------------------"""

#node list for generating figures
cutoff = 2
node_list = [1]
if not do_full:

    node_list.extend(list(range(5,T+5,5)))
if do_full:
    #node_list.extend(range(T+50,num_nodes,50))
    node_list = []
    node_list.extend(list(range(0,110,10)))
    T = num_nodes
    node_list.extend(list(range(150,T,50)))
    node_list.append(T)
    print(node_list)

node_list = node_list[cutoff:]

plotting.get_liftup_graph(node_list= node_list,total_nodes = num_nodes,setting = setting,do_greedy = do_greedy,
                            do_GA = do_GA,do_CELF = do_CELF,do_CFR = do_CFR,do_CFR_heuristic = do_CFR_heuristic,
                            do_random = do_random,do_greedy_simulated = do_greedy_simulated,do_full = do_full,T=T)


plotting.get_similarity_matrix(total_nodes=num_nodes,setting = setting,do_greedy = do_greedy,
                                 do_GA = do_GA,do_CELF = do_CELF,do_CFR = do_CFR,do_CFR_heuristic = do_CFR_heuristic,
                                 do_random = do_random,do_greedy_simulated = do_greedy_simulated,do_full = do_full,my_T=similarity_k)

#Show degree distribution:
if get_degree_distribution:
    plotting.get_degree_dist(dataset = dataset,setting = setting)

