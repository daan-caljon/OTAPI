import torch
import src.utils.utils as utils
import pickle as pkl
import yaml
from src.methods.allocation.utils.allocation_utils import *
import os
from torch_geometric.utils import dense_to_sparse
from src.methods.allocation.utils.Genetic_algorithm import run_GA

def run_allocations_runtime(budgets,dataset,T,do_GA,do_CELF,do_CFR,do_CFR_heuristic,do_greedy,do_random,do_greedy_simulated,
                    w_c,w, w_beta_T2Y,beta0, bias_T2Y, betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,
                    betaNeighborTreatment2Outcome, 
                    betaNoise,setting,do_full=False,num_nodes = 2000):
    if do_full:
        extra = "full_"
    else:
        extra = ""
    run_time_dict = {}
    
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
    test_edge_index = dense_to_sparse(testA)[0]
    test_edge_index = test_edge_index.cuda()
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
   
    zero_predicted = get_sum_predicted_outcome(model_netest,test_edge_index,testA,testX,torch.zeros(testA.shape[0], dtype=torch.float32),POTest)
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
    all_treated_predicted = get_sum_predicted_outcome(model_netest,test_edge_index,testA,testX,torch.ones(testA.shape[0], dtype=torch.float32),POTest)
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
    T = max(max(treatment_amount_list),T)
    run_time_dict["single_discount"] = {}
    run_time_dict["degree"] = {}
    if do_GA:
        run_time_dict["GA"] = {}
    if do_CFR:
        run_time_dict["CFR"] = {}
    if do_CFR_heuristic:
        run_time_dict["CFR_heuristic"] = {}
    if do_random:
        run_time_dict["random"] = {}
    if do_greedy:
        run_time_dict["greedy"] = {}
    if do_greedy_simulated:
        run_time_dict["greedy_simulated"] = {}
    if do_CELF:
        run_time_dict["CELF"] = {}
    
    for b in budgets:
        print("budget",b)
        start_time = time.time()
        print("start single discount")
        single_discount_solution_dict = single_discount(data_params,testX,b,testA,POTest,model_netest)
        end_time = time.time()
        sd_time = end_time - start_time
        run_time_dict["single_discount"][b] = sd_time
        print("single discount time",end_time-start_time)
        single_discount_solution_dict[testA.shape[0]] = [torch.ones(testA.shape[0], dtype=torch.float32),all_treated_predicted,all_treated_outcome]
        single_discount_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]

        my_path = "data/allocations/" + setting + "/"
        if not os.path.exists(my_path):
            os.makedirs(my_path)
        start_time = time.time()
        
        binary_tensor_degree, predicted_outcome_degree, actual_outcome_degree = degree_heuristic(data_params,
                                                                                                testX,b,testA,POTest,model_netest)
        end_time = time.time()
        degree_time = end_time - start_time
        run_time_dict["degree"][b] = degree_time
        print("degree time",end_time-start_time)
        degree_solution_dict[b] = [binary_tensor_degree,predicted_outcome_degree,actual_outcome_degree]

        print("degree done")
        if do_GA:
            hyperparameter_defaults["T"] = b
            #gradually increase num_generations
            #Fewer generations are needed for smaller budgets
            hyperparameter_defaults["num_generations"] = b*37 + 300 
            if b > 100:
                hyperparameter_defaults["num_generations"] = 5000
            hyperparameter_defaults["single_discount_solution"] = single_discount_solution_dict[b][0]
            hyperparameter_defaults["degree_solution"] = binary_tensor_degree
            start_time = time.time()
            binary_tensor_GA, predicted_outcome_GA, actual_outcome_GA = run_GA(hyperparameter_defaults,
                                                                            data_params,testX,test_edge_index,testA,b,POTest,model_netest)
            end_time = time.time()
            GA_time = end_time - start_time
            run_time_dict["GA"][b] = GA_time
            print("GA time",end_time-start_time)

            GA_solution_dict[b] = [binary_tensor_GA,predicted_outcome_GA,actual_outcome_GA]
            print("GA done")
        if do_CFR:
            start_time = time.time()
            binary_tensor_CFR, predicted_outcome_CFR, actual_outcome_CFR = CFR_uplift(data_params,
                                                                                    testX,b,testA,POTest,model_CFR)
            end_time = time.time()
            CFR_time = end_time - start_time
            run_time_dict["CFR"][b] = CFR_time
            print("CFR time",end_time-start_time)
            CFR_solution_dict[b] = [binary_tensor_CFR,predicted_outcome_CFR,actual_outcome_CFR]
        if do_CFR_heuristic:
            start_time = time.time()
            binary_tensor_CFR_heuristic, predicted_outcome_CFR_heuristic, actual_outcome_CFR_heuristic = CFR_heuristic(degree_CFR_heuristic,
                                                                                                                    data_params,testX,b,testA,POTest,model_CFR)
            end_time = time.time()
            CFR_heuristic_time = end_time - start_time
            run_time_dict["CFR_heuristic"][b] = CFR_heuristic_time
            print("CFR_heuristic time",end_time-start_time)
            CFR_heuristic_solution_dict[b] = [binary_tensor_CFR_heuristic,predicted_outcome_CFR_heuristic,actual_outcome_CFR_heuristic]
        if do_random:
            start_time = time.time()
            binary_tensor_random, predicted_outcome_random, actual_outcome_random = random_solution(num_gens_random,
                                                                                                    data_params,testX,b,testA,POTest,model_netest)
            end_time = time.time()
            random_time = end_time - start_time
            run_time_dict["random"][b] = random_time
            print("random time",end_time-start_time)
            random_solution_dict[b] = [binary_tensor_random,predicted_outcome_random,actual_outcome_random]
            print("random done")

        #save the solutions already
        #save degree_solution_dict
        path_degree = my_path + extra + "degree_solution_dict_test.pkl"
        with open(path_degree, 'wb') as f:
            pkl.dump(degree_solution_dict, f)
        
        #save single_discount_solution_dict
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
        #save runtime dict as yaml file
        run_time_dict_path = "data/allocations/" + setting + "/"
        if not os.path.exists(run_time_dict_path):
            os.makedirs(run_time_dict_path)
    
        #add timestamp to the filename
        timestamp = time.strftime("%Y%m%d%H%M%S")
        run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + "_" + timestamp + ".yaml"
        #convert to yaml
        run_time_dict = {str(k): {str(sub_k): v for sub_k, v in sub_dict.items()}
                    for k, sub_dict in run_time_dict.items()}
        with open(run_time_dict_path, 'w') as file:
            yaml.dump(run_time_dict, file, default_flow_style=False)




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

    T = max(budgets)

    if do_greedy:
        start_time = time.time()
        greedy_solution_dict,greedy_time_dict = greedy_heuristic(data_params,testX,T,testA,POTest,model_netest,budgets=budgets)
        end_time = time.time()
        greedy_time = end_time - start_time
        run_time_dict["greedy"] = greedy_time_dict
        print("greedy time",end_time-start_time)

        #save greedy_solution_dict:
        #save greedy_solution_dict_test
        greedy_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
        path_greedy = my_path + extra +"greedy_solution_dict_test.pkl"
        with open(path_greedy, 'wb') as f:
            pkl.dump(greedy_solution_dict, f)
        #update the run_time_dict yaml

        timestamp = time.strftime("%Y%m%d%H%M%S")
        run_time_dict_path = "data/allocations/" + setting + "/"

        if not os.path.exists(run_time_dict_path):
            os.makedirs(run_time_dict_path)
        run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + "_" + timestamp + ".yaml"
        #convert to yaml
        run_time_dict = {str(k): {str(sub_k): v for sub_k, v in sub_dict.items()}
                        for k, sub_dict in run_time_dict.items()}
        with open(run_time_dict_path, 'w') as file:
            yaml.dump(run_time_dict, file, default_flow_style=False)

        
    if do_greedy_simulated:
        start_time = time.time()
        greedy_simulated_solution_dict,greedy_simulated_time_dict = greedy_heuristic(data_params,testX,b,testA,POTest,model_netest,use_simulation=True,budgets=budgets)
        end_time = time.time()
        greedy_simulated_time = end_time - start_time
        run_time_dict["greedy_simulated"] = greedy_simulated_time_dict
        print("greedy simulated time",end_time-start_time)
        #save greedy_solution_dict:
        #save greedy_solution_dict_test
        greedy_simulated_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
        path_greedy_simulated = my_path +extra + "greedy_simulated_solution_dict_test.pkl"
        with open(path_greedy_simulated, 'wb') as f:
            pkl.dump(greedy_simulated_solution_dict, f)
        #update the run_time_dict yaml

        run_time_dict_path = "data/allocations/" + setting + "/"
        if not os.path.exists(run_time_dict_path):
            os.makedirs(run_time_dict_path)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + "_" + timestamp + ".yaml"
        # run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + ".yaml"
        #convert to yaml
        run_time_dict = {str(k): {str(sub_k): v for sub_k, v in sub_dict.items()}
                        for k, sub_dict in run_time_dict.items()}
        with open(run_time_dict_path, 'w') as file:
            yaml.dump(run_time_dict, file, default_flow_style=False)


    if do_CELF:
        start_time = time.time()
        celf_solution_dict,celf_time_dict = run_celf(diffusion_prob_celf,num_simulations_celf,data_params,testX,testA,T,POTest,model_netest,budgets=budgets)
        end_time = time.time()
        celf_time = end_time - start_time
        run_time_dict["CELF"] = celf_time_dict
        print("CELF time",end_time-start_time)
        #save celf_solution_dict:
        #save celf_solution_dict_test
        celf_solution_dict[0] = [torch.zeros(testA.shape[0], dtype=torch.float32),zero_predicted,zero_actual]
        path_celf = my_path + extra +"celf_solution_dict_test.pkl"
        with open(path_celf, 'wb') as f:
            pkl.dump(celf_solution_dict, f)
        #update the run_time_dict yaml
        run_time_dict_path = "data/allocations/" + setting + "/"
        if not os.path.exists(run_time_dict_path):
            os.makedirs(run_time_dict_path)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + "_" + timestamp + ".yaml"
        # run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + ".yaml"
        #convert to yaml
        run_time_dict = {str(k): {str(sub_k): v for sub_k, v in sub_dict.items()}
                        for k, sub_dict in run_time_dict.items()}
        with open(run_time_dict_path, 'w') as file:
            yaml.dump(run_time_dict, file, default_flow_style=False)


    #save run_time dict as yaml file
    run_time_dict_path = "data/allocations/" + setting + "/"
    if not os.path.exists(run_time_dict_path):
        os.makedirs(run_time_dict_path)
    run_time_dict_path = run_time_dict_path + "run_time_dict_" + setting + ".yaml"
    #convert to yaml
    run_time_dict = {str(k): {str(sub_k): v for sub_k, v in sub_dict.items()}
                    for k, sub_dict in run_time_dict.items()}
    with open(run_time_dict_path, 'w') as file:
        yaml.dump(run_time_dict, file, default_flow_style=False)