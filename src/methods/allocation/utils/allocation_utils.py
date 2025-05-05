import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import src.utils.utils as utils
import time as time
# import src.methods.allocation.utils.Genetic_algorithm as ga
import src.methods.allocation.utils.CELF as celf

def predict_outcome(model,edge_index,A,X,T,PO,normalize=False):
    """
    Function to predict the outcome for a treatment vector T
    """  
    model.eval()  
    _,_,pred_outcome,_,_ = model(edge_index.cuda(),A.cuda(), X.cuda(), T.cuda())
    #Not used in paper
    if normalize:
        pred_outcome = utils.PO_normalize_recover(True,PO,pred_outcome)
    return pred_outcome

def sigmod(x):
    """
    Function to calculate the sigmoid of a value x
    """
    return 1/(1 + np.exp(-x))

def potentialOutcomeSimulation(data_params,X,A,T,Z=None,proba = False):
    """
    Function to simulate the potential outcome for a treatment vector T
    Described in the paper
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param A: adjacency matrix
    :param T: treatment vector
    """


    w= np.array(data_params["w"])
    w_beta_T2Y = np.array(data_params["w_beta_T2Y"])
    T = np.array(T,dtype=np.float32)
    epsilon = torch.zeros(T.shape).numpy()
    
    covariate2OutcomeMechanism = np.matmul(w,X.T) #X.T is transpose ---> this used to use sigmod
    neighbors = np.sum(A,1)
    neighborAverage = np.divide(np.matmul(A, covariate2OutcomeMechanism.reshape(-1)), neighbors)
    beta_T2Y = np.matmul(w_beta_T2Y,X.T) + data_params["bias_T2Y"] #Voor nu enkel positieve waarden?
    if Z is None:
        neighborAverageT = np.divide(np.matmul(A, T.reshape(-1)), neighbors)
    else:
        print ("use Z")
        neighborAverageT = np.array(Z)

    total_Treat2Outcome = data_params["betaTreat2Outcome"]*beta_T2Y
    total_network2Outcome = data_params["betaNeighborTreatment2Outcome"]*beta_T2Y
    
    T = np.array(T)
    
    potentialOutcome = data_params["beta0"]+ total_Treat2Outcome*T + data_params["betaCovariate2Outcome"]*covariate2OutcomeMechanism + data_params["betaNeighborCovariate2Outcome"]*neighborAverage+total_network2Outcome*neighborAverageT+data_params["betaNoise"]*epsilon
    
    potentialOutcome = sigmod(potentialOutcome)
   
    return potentialOutcome

def get_sum_potential_outcome(data_params,X,A,T,Z=None,proba = False):
    return np.sum(potentialOutcomeSimulation(data_params,X,A,T,Z,proba))

def get_sum_predicted_outcome(model,edge_index,A,X,T,PO):
    return np.sum(predict_outcome(model,edge_index,A,X,T,PO,normalize=False).detach().cpu().numpy())



def degree_heuristic(data_params,X,T,A,PO,model):
    """
    function that outputs the degree heuristic solution
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :return binary tensor with solution
    :return predicted outcome of the solution (according to the given model)
    :return actual outcome of the solution (according to the simulation)
    """
    edge_index= dense_to_sparse(A)[0].cuda()
    node_degrees = np.sum(A.cpu().numpy(), axis=1)
    top_indices = np.argsort(node_degrees)[-T:]
    binary_tensor = torch.zeros(X.shape[0], dtype=torch.float32)
    binary_tensor[top_indices] = 1
    predicted_outcome = get_sum_predicted_outcome(model,edge_index,A,X,binary_tensor,PO)
    actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),binary_tensor.cpu().numpy())
    return binary_tensor,predicted_outcome, actual_outcome

def greedy_heuristic(data_params,X,T,A,PO,model,use_simulation = False,budgets= None):
    """
    function that outputs the greedy heuristic solution as a dictionary of the iterations
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :param use_simulation: boolean to use the simulatio or the model
    this parameter is used to calculate the upper bound (UB)
    :return dictionary with the solution at each iteration
    """
    if budgets is None:
        budgets = [T]
    budget_dict = {}
    current_budget = budgets[0]
    if len(budgets) > 1:
        budgets = budgets[1:]
    start_time_budget = time.time()
    

    edge_index = dense_to_sparse(A)[0].cuda()
    solution_dict = {}
    current_T = torch.zeros(A.shape[0], dtype=torch.float32)
    current_best = 0
    nodes_treated = []
    for t in range(1,T+1):
        print("start of iteration",t,"of",T)
        start_time= time.time()
        for node in range(current_T.shape[0]):
            if node in nodes_treated:
                continue
            T_temp = current_T.clone()
            T_temp[node] = 1
            if use_simulation:
                current_total = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),T_temp.cpu().numpy())
            else:
                
                current_total = get_sum_predicted_outcome(model,edge_index,A,X,T_temp,PO)
            if current_total > current_best:
                current_best = current_total
                best_node = node
        
        if current_budget == t:
            budget_dict[t] = time.time()-start_time_budget
            current_budget = budgets[0]
            if len(budgets) > 1:
                budgets = budgets[1:]
            print("Budget reached",t)
            print("Budget dict",budget_dict)
        print("iteration",t,"took",time.time()-start_time,"seconds")
        print("Added node",best_node)
        print("Current best",current_best)
        current_T[best_node] = 1
        nodes_treated.append(best_node)
        if not use_simulation:
            current_best_actual = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),current_T.cpu().numpy())
        else:
            current_best_actual = current_best
        solution_dict[t] = [nodes_treated.copy(),current_best, current_best_actual]
    return solution_dict,budget_dict
    

def single_discount(data_params,X,T,A,POTrain,model):
    """
    function that outputs the single discount solution as a dictionary of the iterations
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :return dictionary with the solution at each iteration
    """
    edge_index = dense_to_sparse(A)[0].cuda()
    node_degrees = np.sum(A.cpu().numpy(), axis=1)
    indices_degree = list(np.argsort(node_degrees))
    binary_tensor = torch.zeros(A.shape[0], dtype=torch.float32)
    A_clone = A.clone()
    discount_dict = {}
    
    degrees_zero = False
    for t in range(1,T+1):
        # Calculate the degree of each node by summing along the rows
        #At some point degrees will be 0 for all nodes, then we just add nodes with highest degree remaining
        if not degrees_zero:
            degrees = torch.sum(A_clone, dim=1)

            # Find the index with the highest degree
            index_highest_degree = torch.argmax(degrees).item()
        if index_highest_degree in indices_degree:

            indices_degree.remove(index_highest_degree)
        else:
            degrees_zero = True
            #index_highest_degree = all_indices_list[0]
            index_highest_degree = indices_degree[0]
            #all_indices_list.remove(index_highest_degree)
            indices_degree.remove(index_highest_degree)
        binary_tensor[index_highest_degree] = 1
        if not degrees_zero:
            A_clone[index_highest_degree,:] = 0
            A_clone[:,index_highest_degree] = 0
    
        actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),binary_tensor.cpu().numpy())
        predicted_outcome = get_sum_predicted_outcome(model,edge_index,A,X,binary_tensor,POTrain)
        discount_dict[t] = [binary_tensor.clone(),predicted_outcome,actual_outcome]
    return discount_dict

def random_solution(num_gens,data_params,X,T,A,POTrain,model):
    """
    function that outputs the random solution for a given T
    :param num_gens: number of simulations
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :return binary tensor with solution
    :return predicted outcome of the solution (according to the given model)
    :return actual outcome of the solution (according to the simulation)"""
    edge_index = dense_to_sparse(A)[0].cuda()
    total_actual = 0
    total_predicted = 0
    for _ in range(num_gens):
        binary_tensor = torch.zeros(A.shape[0], dtype=torch.float32)
        binary_tensor[torch.randperm(A.shape[0])[:T]] = 1
        #print("binary_check",torch.sum(binary_tensor))
        actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),binary_tensor.cpu().numpy())
        predicted_outcome = get_sum_predicted_outcome(model,edge_index,A,X,binary_tensor,POTrain)
        total_actual += actual_outcome
        total_predicted += predicted_outcome
    return binary_tensor,total_predicted/num_gens,total_actual/num_gens

def CFR_uplift(data_params,X,T,A,POTrain,model):
    """
    function that outputs the CFR solution for a given T
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :return binary tensor with solution
    :return predicted outcome of the solution (according to the given model)
    :return actual outcome of the solution (according to the simulation)"""
    edge_index = dense_to_sparse(A)[0].cuda()
    zeroTreat = predict_outcome(model,edge_index,A,X,torch.zeros(A.shape[0], dtype=torch.float32),POTrain,normalize=False).detach().cpu().numpy()
    allTreat = predict_outcome(model,edge_index,A,X,torch.ones(A.shape[0], dtype=torch.float32),POTrain,normalize=False).detach().cpu().numpy()
    diff = allTreat - zeroTreat
    indices = np.argsort(diff)[-T:]
    binary_tensor = torch.zeros(A.shape[0], dtype=torch.float32)
    binary_tensor[indices] = 1
    actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),binary_tensor.cpu().numpy())
    predicted_outcome = get_sum_predicted_outcome(model,edge_index,A,X,binary_tensor,POTrain)
    return binary_tensor,predicted_outcome,actual_outcome

def CFR_heuristic(degree_amount,data_params,X,T,A,POTrain,model):
    """
    function that outputs the CFR heuristic solution for a given T
    this heuristic selects the top degree nodes and then selects the nodes with the highest uplift
    :param degree_amount: number of top degree nodes to select
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :return binary tensor with solution
    :return predicted outcome of the solution (according to the given model)
    :return actual outcome of the solution (according to the simulation)"""
    edge_index = dense_to_sparse(A)[0].cuda()
    node_degrees = np.sum(A.cpu().numpy(), axis=1)
    if degree_amount > T:
        degree_amount = T
    top_indices = np.argsort(node_degrees)[-degree_amount:]
    zeroTreat = predict_outcome(model,A,X,torch.zeros(A.shape[0], dtype=torch.float32),POTrain,normalize=False).detach().cpu().numpy()
    allTreat = predict_outcome(model,A,X,torch.ones(A.shape[0], dtype=torch.float32),POTrain,normalize=False).detach().cpu().numpy()
    diff = allTreat - zeroTreat
    binary_tensor = torch.zeros(A.shape[0], dtype=torch.float32)
    binary_tensor[top_indices] = 1
    #This way the top degree nodes wont be selected by CFR heuristic
    diff[top_indices] = -1
    if degree_amount < T:
        indices = np.argsort(diff)[-T+degree_amount:]
        binary_tensor[indices] = 1
    actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),binary_tensor.cpu().numpy())
    predicted_outcome = get_sum_predicted_outcome(model,edge_index,A,X,binary_tensor,POTrain)
    return binary_tensor,predicted_outcome,actual_outcome

# def run_GA(hyperparameter_defaults,data_params,X,A,T,POTrain,model):
#     """Run the genetic algorithm for a given T
#     :param hyperparameter_defaults: dictionary with the hyperparameters of the genetic algorithm
#     :param data_params: dictionary with the parameters of the simulation
#     :param X: covariate matrix
#     :param T: number of treatments
#     :param A: adjacency matrix
#     :param PO: potential outcome
#     :param model: causal model
#     :return binary tensor with solution
#     :return predicted outcome of the solution (according to the given model)
#     :return actual outcome of the solution (according to the simulation)"""

#     GA = ga.GeneticAlgorithm(model,A,X,POTrain,hyperparameter_defaults)
#     solution_ga, solution_fitness_ga = GA.run()
#     actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),solution_ga)
#     predicted_outcome = get_sum_predicted_outcome(model,A,X,torch.Tensor(solution_ga),POTrain)
#     return solution_ga,predicted_outcome,actual_outcome

def run_celf(diffusion_prob,num_simulations,data_params,X,A,T,POTrain,model,budgets=None):
    """Run the CELF algorithm until the budget T is reached
    :param diffusion_prob: diffusion probability
    :param num_simulations: number of MC simulations
    :param data_params: dictionary with the parameters of the simulation
    :param X: covariate matrix
    :param T: number of treatments (budget)
    :param A: adjacency matrix
    :param PO: potential outcome
    :param model: causal model
    :return dictionary with the solution at each iteration"""
    if budgets is None:
        budgets = [T]
    edge_index = dense_to_sparse(A)[0].cuda()
    _, celf_T_test,celf_solution_test,celf_solution_dict_IC, _,_,time_dict = celf.CELF(A.cpu().numpy(),diffusion_prob,T,num_simulations,budgets=budgets)
    celf_solution_dict = {}
    for t,solution in celf_solution_dict_IC.items():
        print(t,solution)
        solution_tensor = torch.zeros(A.shape[0], dtype=torch.float32)
        solution_tensor[solution] = 1
        print("solution_tensor",solution_tensor.sum())
        actual_outcome = get_sum_potential_outcome(data_params,X.cpu().numpy(),A.cpu().numpy(),solution_tensor.cpu().numpy())
        predicted_outcome = get_sum_predicted_outcome(model,edge_index,A,X,solution_tensor,POTrain)
        celf_solution_dict[t] = [solution,predicted_outcome,actual_outcome]

    return celf_solution_dict,time_dict
