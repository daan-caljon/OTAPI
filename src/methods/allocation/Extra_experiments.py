import torch
import time
import utils.utils as utils
import numpy as np
from src.methods.causal_models.model import NetEstimator
from src.methods.causal_models.baselineModels import GCN_DECONF,CFR,GCN_DECONF_INTERFERENCE,CFR_INTERFERENCE
from src.methods.causal_models.experiment import Experiment
import argparse
import pickle as pkl
import src.methods.allocation.Genetic_algorithm as ga
import src.methods.allocation.CELF as celf
from src.methods.allocation.Predictions import *
import matplotlib
def run_extra_allocations(dataset,T,do_GA,do_CELF,do_CFR,do_CFR_heuristic,do_greedy,do_random,do_greedy_simulated,
                    w_c,w, w_beta_T2Y,beta0, bias_T2Y, betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,
                    betaNeighborTreatment2Outcome, 
                    betaNoise,setting,node_list = None,do_full=False,num_nodes = 2000):
    
    cuda=True

    z_11 = 0.7
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



    # dataset = args.dataset

    file = "data/simulated/" + setting +".pkl"
    with open(file,"rb") as f:
            data = pkl.load(f)
    dataTrain,dataVal,dataTest = data["train"],data["val"],data["test"]
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain = utils.dataTransform(dataTrain,cuda)
    # valA, valX, valT,cfValT,POVal,cfPOVal = utils.dataTransform(dataVal,cuda)
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
    hyperparameter_defaults["keep_parents"] = 6
    hyperparameter_defaults["mutation_num_genes"] = 1 #--> 1 gives best results
    hyperparameter_defaults["degree_init"] = True
    hyperparameter_defaults["draw_graph"] = False
    hyperparameter_defaults["mutation_type"] = "random"
    hyperparameter_defaults["crossover_type"] = "uniform" #--> uniform worked best
    hyperparameter_defaults["parent_selection_type"] = "sss"
    hyperparameter_defaults["keep_elitism"] = 5

    num_gens_random = 100
    diffusion_prob_celf = 0.01
    num_simulations_celf = 1000 
    degree_CFR_heuristic = 20
    shape = testA.shape
    print("total nodes",shape)
    num_nodes = shape[0]
    # if dataset == "full_sim":
    #     large_amount_nodes = [150,200,250,300,350,400,450,500,750,1000,1500,2000,2500,3000,3500,4000,4500]
    # elif dataset == "BC":
    #     large_amount_nodes = [150,200,250,300,350,400,450,500,750,1000,1250,1500,1784]
    #get values for zero
    
    degree_solution_dict = {}
    single_discount_solution_dict = {}
    GA_solution_dict = {}
    random_solution_dict = {}
    CFR_solution_dict = {}

    T = max(node_list) +1
    treatment_amount_list = node_list
    single_discount_solution_dict = single_discount(data_params,testX,T,testA,POTest,model_netest)


    print(treatment_amount_list)
    print("T",T)
    for t in treatment_amount_list:
        print("iteration",t)
        
        binary_tensor_degree, predicted_outcome_degree, actual_outcome_degree = degree_heuristic(data_params,
                                                                                                testX,t,testA,POTest,model_netest)
        # binary_tensor_single_discount, predicted_outcome_single_discount, actual_outcome_single_discount = single_discount(data_params,
        #                                                                                                                 testX,t,testA,POTest,model_netest)
        degree_solution_dict[t] = [binary_tensor_degree,predicted_outcome_degree,actual_outcome_degree]
        #single_discount_solution_dict[t] = [binary_tensor_single_discount,predicted_outcome_single_discount,actual_outcome_single_discount]
        print("degree done")
        if do_GA:
            hyperparameter_defaults["T"] = t
            #gradually increase num_generations for runtime reasons
            hyperparameter_defaults["num_generations"] = t*37 + 300 #CHANGE BACK!!!!
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
        if do_random:
            binary_tensor_random, predicted_outcome_random, actual_outcome_random = random_solution(num_gens_random,
                                                                                                    data_params,testX,t,testA,POTest,model_netest)
            random_solution_dict[t] = [binary_tensor_random,predicted_outcome_random,actual_outcome_random]
            print("random done")



    #save all dicts:
    #save degree_solution_dict_test
    my_path = "data/allocations/" + setting + "/"
    extra = "extra_experiment_"
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
    if do_random:
        #save random_solution_dict_test
        path_random = my_path + extra +"random_solution_dict_test.pkl"
        with open(path_random,"wb") as f:
            pkl.dump(random_solution_dict,f)

