import torch
import time
import utils.utils as utils
import numpy as np
from src.methods.causal_models.model import NetEstimator
from  src.methods.causal_models.baselineModels import GCN_DECONF,CFR,GCN_DECONF_INTERFERENCE,CFR_INTERFERENCE
from  src.methods.causal_models.experiment import Experiment
import argparse
import pickle as pkl
import src.methods.allocation.Genetic_algorithm as ga
import src.methods.allocation.CELF as celf
from src.methods.allocation.Predictions import *
import os

def run_allocations(dataset,T,do_GA,do_CELF,do_CFR,do_CFR_heuristic,do_greedy,do_random,do_greedy_simulated,
                    w_c,w, w_beta_T2Y,beta0, bias_T2Y, betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,
                    betaNeighborTreatment2Outcome, 
                    betaNoise,setting,do_full=False,num_nodes = 2000):
   
    cuda=True

    z_11 = 0.7 #This is used if you want to evaluate total effect of treatment (not necessary most of the time)
    z_22 = 0.2

    data_params = {}
    data_params["z_11"] = z_11
    data_params["z_22"] = z_22
    data_params["betaTreat2Outcome"] = betaTreat2Outcome
    data_params["bias_T2Y"] = bias_T2Y
    data_params["betaCovariate2Outcome"] = betaCovariate2Outcome
    data_params["betaNeighborTreatment2Outcome"] = betaNeighborTreatment2Outcome
    data_params["betaNeighborCovariate2Outcome"] = betaNeighborCovariate2Outcome
    data_params["betaNoise"] = betaNoise
    data_params["beta0"] = beta0
    data_params["w_c"] = w_c
    data_params["w"] = w
    data_params["w_beta_T2Y"] = w_beta_T2Y

    file = "data/simulated/" + setting +".pkl"
    with open(file,"rb") as f:
            data = pkl.load(f)
    #Only use test data
    _,_,dataTest = data["train"],data["val"],data["test"]
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    testA, testX, testT,cfTestT,POTest,cfPOTest = utils.dataTransform(dataTest,cuda)

    #Load in NetEst model and CFR
    model_netest_path = "models/"+setting+"/"+"NetEstimator" + "_"+ setting+".pth"
    model_CFR_path =  "models/"+setting+"/"+"CFR" + "_"+ setting+".pth"
  
    model_netest = torch.load(model_netest_path)
    model_CFR = torch.load(model_CFR_path)
    model_netest.eval()
    model_CFR.eval()
    epsilon = torch.zeros(testT.shape).numpy()
    model_netest = model_netest.cuda()
    model_CFR = model_CFR.cuda()
    #Genetic algo
    hyperparameter_defaults = dict()
    hyperparameter_defaults["T"] = T
    hyperparameter_defaults["num_generations"] = 4000
    hyperparameter_defaults["num_parents_mating"] = 15
    hyperparameter_defaults["sol_per_pop"] = 40
    hyperparameter_defaults["keep_parents"] = 5
    hyperparameter_defaults["mutation_num_genes"] = 1 #--> 1 gives best results
    hyperparameter_defaults["degree_init"] = True
    hyperparameter_defaults["draw_graph"] = False
    hyperparameter_defaults["mutation_type"] = "random"
    hyperparameter_defaults["crossover_type"] = "uniform" #--> uniform worked best
    hyperparameter_defaults["parent_selection_type"] = "sss"
    hyperparameter_defaults["keep_elitism"] = 5

    num_gens_random = 100 #number of simulations for random solution
    diffusion_prob_celf = 0.01 #probability in IC model
    num_simulations_celf = 1000 #MC simulation
    degree_CFR_heuristic = 20 #heuristic where we first treat the 20 nodes with highest degree and then use TARNet/CFR
    shape = testA.shape
    print("total nodes",shape)
    num_nodes = shape[0]
   
    zero_predicted = get_sum_predicted_outcome(model_netest,testA,testX,torch.zeros(testA.shape[0], dtype=torch.float32),POTest)
    zero_actual = get_sum_potential_outcome(data_params,testX.cpu().numpy(),testA.cpu().numpy(),torch.zeros(testA.shape[0], dtype=torch.float32).cpu().numpy())

   
    degree_solution_dict = {}

    random_solution_dict = {}
    CFR_solution_dict = {}
    CFR_heuristic_solution_dict = {}
    GA_solution_dict = {}
    degree_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
    random_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
    CFR_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
    CFR_heuristic_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
    GA_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]


    #Define the budgets for which to save to do the allocations
    treatment_amount_list = list(range(1,21))
    treatment_amount_list.extend(list(range(25,T+5,5)))

    #Actual and predicted outcomes for treating all nodes
    all_treated_outcome = get_sum_potential_outcome(data_params,testX.cpu().numpy(),testA.cpu().numpy(),torch.ones(testA.shape[0], dtype=torch.float32).cpu().numpy())
    all_treated_predicted = get_sum_predicted_outcome(model_netest,testA,testX,torch.ones(testA.shape[0], dtype=torch.float32),POTest)
    CFR_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]
    CFR_heuristic_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]
    GA_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]
    random_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]
    degree_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]


    #Do full means that the all budgets are calculated (not feasible for large networks)
    if do_full:
        treatment_amount_list = [1]
        treatment_amount_list.extend(list(range(5,105,5)))
        T = num_nodes
        treatment_amount_list.extend(list(range(150,T,50)))
        treatment_amount_list.append(T)
        if dataset == "Flickr":
            treatment_amount_list.extend([118,472,943]) #to get the 5,10, 20% of nodes

    single_discount_solution_dict = single_discount(data_params,testX,T,testA,POTest,model_netest)
    single_discount_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]
    single_discount_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]

    print(treatment_amount_list)
    print("T",T)


    for t in treatment_amount_list:
        print("iteration",t)
        
        binary_tensor_degree, predicted_outcome_degree, actual_outcome_degree = degree_heuristic(data_params,
                                                                                                testX,t,testA,POTest,model_netest)
        degree_solution_dict[t] = [binary_tensor_degree,predicted_outcome_degree,actual_outcome_degree]
        print("degree done")
        if do_GA:
            hyperparameter_defaults["T"] = t
            #gradually increase num_generations
            #Fewer generations are needed for smaller budgets
            hyperparameter_defaults["num_generations"] = t*37 + 300 
            if t > 100:
                hyperparameter_defaults["num_generations"] = 5000
            hyperparameter_defaults["single_discount_solution"] = single_discount_solution_dict[t][0]
            hyperparameter_defaults["degree_solution"] = binary_tensor_degree
            binary_tensor_GA, predicted_outcome_GA, actual_outcome_GA = run_GA(hyperparameter_defaults,
                                                                            data_params,testX,testA,t,POTest,model_netest)
            GA_solution_dict[t] = [binary_tensor_GA,predicted_outcome_GA,actual_outcome_GA]
            print("GA done")
        if do_CFR:
            binary_tensor_CFR, predicted_outcome_CFR, actual_outcome_CFR = CFR_uplift(data_params,
                                                                                    testX,t,testA,POTest,model_CFR)
            CFR_solution_dict[t] = [binary_tensor_CFR,predicted_outcome_CFR,actual_outcome_CFR]
        if do_CFR_heuristic:
            binary_tensor_CFR_heuristic, predicted_outcome_CFR_heuristic, actual_outcome_CFR_heuristic = CFR_heuristic(degree_CFR_heuristic,
                                                                                                                    data_params,testX,t,testA,POTest,model_CFR)
            CFR_heuristic_solution_dict[t] = [binary_tensor_CFR_heuristic,predicted_outcome_CFR_heuristic,actual_outcome_CFR_heuristic]
        if do_random:
            binary_tensor_random, predicted_outcome_random, actual_outcome_random = random_solution(num_gens_random,
                                                                                                    data_params,testX,t,testA,POTest,model_netest)
            random_solution_dict[t] = [binary_tensor_random,predicted_outcome_random,actual_outcome_random]
            print("random done")



    #save all dicts:
    #save degree_solution_dict_test
    my_path = "data/allocations/" + setting + "/"
    if not os.path.exists(my_path):
        os.makedirs(my_path)

    if do_full:
        extra = "full_"
    else:
        extra = ""
    path_degree = my_path + extra + "degree_solution_dict_test.pkl"
    with open(path_degree, 'wb') as f:
        pkl.dump(degree_solution_dict, f)

    path_single_discount = my_path + extra +"single_discount_solution_dict_test.pkl"
    with open(path_single_discount,"wb") as f:
        pkl.dump(single_discount_solution_dict,f)
    if do_GA:
        #save GA_solution_dict_test
        path_GA = my_path + extra +"GA_solution_dict_test.pkl"
        with open(path_GA,"wb") as f:
            pkl.dump(GA_solution_dict,f)
    if do_CFR:
        #save CFR_solution_dict_test
        path_CFR = my_path + extra +"CFR_solution_dict_test.pkl"
        with open(path_CFR,"wb") as f:
            pkl.dump(CFR_solution_dict,f)
    if do_CFR_heuristic:
        #save CFR_heuristic_solution_dict_test
        path_CFR_heuristic = my_path + extra +"CFR_heuristic_solution_dict_test.pkl"
        with open(path_CFR_heuristic,"wb") as f:
            pkl.dump(CFR_heuristic_solution_dict,f)
    if do_random:
        #save random_solution_dict_test
        path_random = my_path + extra +"random_solution_dict_test.pkl"
        with open(path_random,"wb") as f:
            pkl.dump(random_solution_dict,f)

    if do_greedy:
        greedy_solution_dict = greedy_heuristic(data_params,testX,T,testA,POTest,model_netest)
        #save greedy_solution_dict:
        #save greedy_solution_dict_test
        greedy_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
        path_greedy = my_path + extra +"greedy_solution_dict_test.pkl"
        with open(path_greedy, 'wb') as f:
            pkl.dump(greedy_solution_dict, f)

    if do_greedy_simulated:
        greedy_simulated_solution_dict = greedy_heuristic(data_params,testX,T,testA,POTest,model_netest,use_simulation=True)
        #save greedy_solution_dict:
        #save greedy_solution_dict_test
        greedy_simulated_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
        path_greedy_simulated = my_path +extra + "greedy_simulated_solution_dict_test.pkl"
        with open(path_greedy_simulated, 'wb') as f:
            pkl.dump(greedy_simulated_solution_dict, f)

    if do_CELF:
        celf_solution_dict = run_celf(diffusion_prob_celf,num_simulations_celf,data_params,testX,testA,T,POTest,model_netest)
        #save celf_solution_dict:
        #save celf_solution_dict_test
        celf_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
        path_celf = my_path + extra +"celf_solution_dict_test.pkl"
        with open(path_celf, 'wb') as f:
            pkl.dump(celf_solution_dict, f)

