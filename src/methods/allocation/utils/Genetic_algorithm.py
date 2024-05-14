import time
import numpy as np
import pygad
import torch
from methods.allocation.utils.allocation_utils import *



class GeneticAlgorithm():
    """
    Class to run the Genetic Algorithm using pygad
    https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
    """
    def __init__(self,model,A,X,POTrain,config):  
        self.T = config["T"]
        self.num_generations = config["num_generations"]
        self.num_parents_mating = config["num_parents_mating"]
        self.sol_per_pop = config["sol_per_pop"]
        self.keep_parents = config["keep_parents"]
        self.mutation_num_genes = config["mutation_num_genes"]
        self.degree_init = config["degree_init"]
        self.mutation_type = config["mutation_type"]
        self.crossover_type = config["crossover_type"]
        self.parent_selection_type = config["parent_selection_type"]
        self.keep_elitism = config["keep_elitism"]
        self.degree_solution = config["degree_solution"]
        self.single_discount_solution = config["single_discount_solution"]
        self.A = A
        self.model = model
        self.X = X
        self.POTrain= POTrain

        self.num_genes = len(self.A)

        self.init_range_low = 0
        self.init_range_high = 1
        #All elements in the allocation set can be either 0 or 1
        self.gene_space = [0,1]

        #Specify whether to use the degree and SD heuristic solutions in the initial population
        if self.degree_init:

            self.initial_population = [gen_init(self.A,self.T).numpy() for _ in range(self.sol_per_pop-2)]
            self.initial_population.append(self.degree_solution.numpy())
            self.initial_population.append(self.single_discount_solution.numpy())

        else: 
            self.initial_population = [list(gen_init(self.A,self.T)) for _ in range(self.sol_per_pop)]


    def fitness_func(self,ga_instance, solution, solution_idx):
        """
        Function to calculate the fitness of a solution
        This function will be called by the Genetic Algorithm"""
        #Budget constraint
        if np.sum(solution) > self.T:
            return 0  
        #Calculate the fitness using causal model
        my_fitness = get_sum_predicted_outcome(self.model,self.A.cuda(),self.X.cuda(),torch.Tensor(solution).cuda(),self.POTrain)

        return my_fitness

    
    def run(self):

        self.ga_instance =  pygad.GA(num_generations=self.num_generations,
                       num_parents_mating=self.num_parents_mating,
                       fitness_func=self.fitness_func,
                       sol_per_pop=self.sol_per_pop,
                       num_genes=self.num_genes,
                       init_range_low=self.init_range_low,
                       init_range_high=self.init_range_high,
                       parent_selection_type=self.parent_selection_type,
                       keep_parents=self.keep_parents,
                       crossover_type=self.crossover_type,
                       mutation_type=self.mutation_type,
                       mutation_num_genes=self.mutation_num_genes,
                       gene_space= self.gene_space,
                       initial_population=self.initial_population,
                       #on_generation = self.log_function,
                       mutation_by_replacement= True,
                       keep_elitism= self.keep_elitism)
        print("Started")
        self.start_time = time.time()
        self.ga_instance.run()
        self.end_time = time.time()
        print("Time elapsed:", round(self.end_time-self.start_time))

        self.solution, self.solution_fitness, self.solution_idx = self.ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=self.solution_fitness))
        return self.solution, self.solution_fitness

def gen_init(A,T):
    length = A.shape[0]
    random_tensor = torch.zeros(length)
    indices = torch.randperm(length)[:T]
    random_tensor.view(-1)[indices] = 1
    return random_tensor
        