import numpy as np
import random
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.utils import multi_runs
import time
import heapq
#use NDlib for IC model simulations: https://ndlib.readthedocs.io/en/latest/overview.html


"""Explanation of algorithm:
    1. Calculate the marginal gain of each node
    2. Pick the node with the highest marginal gain
    3. Add it to the solution
    4. While the solution is not complete:
        Recalculate marginal gain for node with highest marginal gain according to the priority queue (heap)
        If this is still the largest marginal gain compared to all other nodes (which were calculated in a previous step)
            Add this node to the solution
        If not: repeat
    5. Return solution

    We also check if the node with the highest marginal gain at that point has already done MC simulations

    """
def CELF(A,p,T,num_simulations=1000):
    """
    Function to run the CELF algorithm
    :param A: adjacency matrix
    :param p: probability of influence
    :param T: number of treatments (budget)
    :param num_simulations: number of MC simulations
    :return: solution, binary array, current influence, solution dictionary, spread list, time list
    solution dictionary contains the solution at each iteration
    """
    solution_dict = {}
    spread_list = []
    time_list = []
    G = nx.from_numpy_array(A)
    start_time=time.time()
    pq = [(-simulate_influence(G, [node],p,num_simulations),node,1) for node in G.nodes()]
    end_time = time.time()
    time_list.append(end_time - start_time)
    print("Iteration 1",time_list[-1])
    # for  node in ic_model.graph.nodes():
    #     current_influence = simulate_influence(G,ic_model, [node], 1000)
    #     heapq.heappush(pq, (-current_influence, node, 1))
    #print(pq)
    heapq.heapify(pq)
    
    current_influence, node,_ = heapq.heappop(pq)
    solution = []
    solution.append(node)
    solution_dict[len(solution)] = solution.copy()
    spread_list.append(-current_influence)
    heapq.heapify(pq)
    while len(solution) < T:
        start_time = time.time()
        check = False
        _, node,_ = heapq.heappop(pq)
        heapq.heapify(pq)
        while not check:
            #_, node = heapq.heappop(pq)
            
            #print("node",node)
            current_influence_difference = (simulate_influence(G, solution + [node],p,num_simulations=num_simulations)- current_influence)
            heapq.heappush(pq, (-current_influence_difference, node,len(solution)+1))
            heapq.heapify(pq)
            _, new_node,last_iteration_calculated = heapq.heappop(pq)
            heapq.heapify(pq)
            #print("current iteration",len(solution)+1,last_iteration_calculated)
            if new_node == node or last_iteration_calculated == len(solution) + 1:
                check = True
                node = new_node
            else:
                node = new_node
        #print("pq",pq)
        current_influence += current_influence_difference
        solution.append(node)
        print("solution",solution)
        spread_list.append(current_influence)
        #print("spread_list",spread_list)
        time_list.append(time.time() - start_time)
        print("Iteration", last_iteration_calculated, ":",time_list[-1])
        solution_dict[len(solution)] = solution.copy() # is this the problem? --> make it not a list?
    zeros_array = np.zeros_like(A[0],dtype=np.float32)
    for node in solution:
        zeros_array[node] = 1
    binary_array = zeros_array
    return solution, binary_array,current_influence, solution_dict,spread_list, time_list


#MC simulation

def simulate_influence(g,seed_nodes,p,num_simulations=1000):
    """
    Function to simulate the influence spread in a network using the Independent Cascade model
    :param g: networkx graph
    :param seed_nodes: list of seed nodes (nodes that are initially treated)
    :param p: probability of influence
    :param num_simulations: number of simulations
    :return: average influence spread"""
    if len(seed_nodes)==1:
        print(seed_nodes)
    num_runs = num_simulations
    config = mc.Configuration()
    config.add_model_initial_configuration('Infected', seed_nodes)
    for e in g.edges():
        config.add_edge_configuration("threshold", e, p)
    # Execute the model
    sum_spread = 0 
    for run in range(num_runs):
        ic_model = ep.IndependentCascadesModel(g)
        ic_model.set_initial_status(config)
        infected = 1
        while infected > 0:
            iterations = ic_model.iteration()
            
            infected = iterations['node_count'][1]
            
        
        sum_spread += iterations['node_count'][2]
    # Get the number of influenced nodes

    return sum_spread/num_runs


