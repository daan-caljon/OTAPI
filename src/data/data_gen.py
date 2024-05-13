import random
import numpy as np
import networkx as nx
import torch
import pickle as pkl
from src.data.datatools import *



def treatmentSimulation(w_c,X,A,betaConfounding,betaNeighborConfounding):

    covariate2TreatmentMechanism = sigmod(np.matmul(w_c,X.T))
    neighbors = np.sum(A,1)
    print (np.sum(neighbors==0))
    neighborAverage = np.divide(np.matmul(A, covariate2TreatmentMechanism.reshape(-1)), neighbors)
    print (np.mean(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage),np.std(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage))
    propensityT= sigmod(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage)
    random_values = np.random.rand(len(propensityT))
    T = (random_values < np.array(propensityT)).astype(int)
    print ("Lenght of treatment vector:{}".format(len(T)))
    mean_T = np.mean(T)
    return T, mean_T


def flipTreatment(T,rate):
    
    numToFlip = int(len(T)*rate)
    nodesToFlip = set(np.random.choice(len(T), numToFlip, replace=False))
    cfT = np.array([1-T[i] if i in nodesToFlip else T[i] for i in range(len(T))])
    
    return cfT,nodesToFlip


def sigmod(x):
    return 1/(1 + np.exp(-x))

def potentialOutcomeSimulation(w,X,A,T,beta0, w_beta_T2Y, bias_T2Y,epsilon,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=None,proba = False):
    w= np.array(w)
    w_beta_T2Y = np.array(w_beta_T2Y)
    T = np.array(T,dtype=np.float32)
    
    
    covariate2OutcomeMechanism = np.matmul(w,X.T) #X.T is transpose 
    neighbors = np.sum(A,1)
    neighborAverage = np.divide(np.matmul(A, covariate2OutcomeMechanism.reshape(-1)), neighbors)
    beta_T2Y = np.matmul(w_beta_T2Y,X.T) + bias_T2Y #Voor nu enkel positieve waarden?
    print("beta_T2Y",beta_T2Y.mean(),beta_T2Y.std())
    if Z is None:
        neighborAverageT = np.divide(np.matmul(A, T.reshape(-1)), neighbors)
    else:
        print ("use Z")
        neighborAverageT = np.array(Z)
   
    total_Treat2Outcome = betaTreat2Outcome*beta_T2Y
    total_network2Outcome = betaNeighborTreatment2Outcome*beta_T2Y
    
    T = np.array(T)
    potentialOutcome = beta0+ total_Treat2Outcome*T + betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage+total_network2Outcome*neighborAverageT+betaNoise*epsilon
    print("potentialOutcome",potentialOutcome.mean(),potentialOutcome.std())
    potentialOutcome = sigmod(potentialOutcome)
    print("proba:", potentialOutcome[1000:1020])
    print("proba mean:", np.mean(np.array(potentialOutcome)))
    print("sample potential outcome")
    if not proba:
        random_values = np.random.rand(len(potentialOutcome))
        print("random values",random_values[1000:1020])
        potentialOutcome = (random_values < np.array(potentialOutcome)).astype(int)
    print(potentialOutcome)
    return potentialOutcome




def generate_data(dataset,w_c,w, w_beta_T2Y,beta0, betaConfounding,betaNeighborConfounding, 
                  bias_T2Y,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, 
                  betaNoise,gen_type,nodes = 2000,edges_new_node = 2, covariate_dim = 10,flipRate = 1,watts_strogatz = False):

    z_11 = 0.7
    z_22 = 0.2
    if dataset == "full_sim":
        if watts_strogatz:
            G = nx.connected_watts_strogatz_graph(nodes, 4, 0.1)
        else:
            G = nx.barabasi_albert_graph(nodes, edges_new_node)
        print("avg_degree",sum(dict(G.degree()).values()) / len(G))
        adj_matrix = nx.adjacency_matrix(G)
    else:
        
        data,parts = readData(dataset)
        trainIndex,valIndex,testIndex = dataSplit(parts)
        trainX, valX, testX = covariateTransform(data,covariate_dim,trainIndex,valIndex,testIndex)
        #We will normalize the covariates over the columns:
        mean_trainX = np.mean(trainX,axis=(0))
        std_trainX = np.std(trainX,axis=(0))
        trainX = (trainX-mean_trainX)/std_trainX
        valX = (valX-np.mean(valX,axis=0))/np.std(valX,axis=0)
        testX = (testX-np.mean(testX,axis=0))/np.std(testX,axis=0)
        print("trainX",trainX)
        trainA, valA, testA = adjMatrixSplit(data,trainIndex,valIndex,testIndex,dataset)

    # Convert the sparse matrix to a dense NumPy array
    if dataset == "full_sim":
        dense_adj_matrix = adj_matrix.toarray()
        A = dense_adj_matrix
    else:
        if gen_type == "train":
            dense_adj_matrix = trainA
            A = dense_adj_matrix
        elif gen_type == "val":
            dense_adj_matrix = valA
            A = dense_adj_matrix
        elif gen_type == "test":
            dense_adj_matrix = testA
            A = dense_adj_matrix
    # Convert the NumPy array to a PyTorch tensor
        tensor_adj_matrix = torch.tensor(dense_adj_matrix, dtype=torch.float32)

    print("adj_matrix",dense_adj_matrix.shape)

    #Generate the covariates
    if dataset == "full_sim":

        X = np.random.randn(dense_adj_matrix.shape[1], covariate_dim)
    else:
        if gen_type == "train":
            #column normalization --> calculate sd and mean for train and then use this also for test?! 
            X = trainX
            
        elif gen_type == "val":
            X = valX
        elif gen_type == "test":
            X = testX
    
    print("X",X)
    

    epsilon = np.random.normal(0,1,X.shape[0])
 
    T, meanT = treatmentSimulation(np.array(w_c),np.array(X),np.array(A),betaConfounding,betaNeighborConfounding) #effect of X to T
    T = torch.tensor(T, dtype=torch.float32)

    PO = potentialOutcomeSimulation(w,X,A,T,beta0,w_beta_T2Y,bias_T2Y,epsilon,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)
    print("PO",PO.shape[0], PO)
    print("PO mean",np.mean(np.array(PO)))
    print("PO std",np.std(np.array(PO)))


    print(T)
    cfT_train,nodesToFlipTrain = flipTreatment(T,flipRate)
    print(cfT_train)
    cfT_train = torch.from_numpy(cfT_train)
    epsiloncf = np.random.normal(0,1,X.shape[0])

    cfPOTrain = potentialOutcomeSimulation(w,X,A,cfT_train,beta0,w_beta_T2Y,bias_T2Y,epsiloncf,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise)


    num = X.shape[0]
    t_1s = torch.ones(num)
    t_0s = torch.zeros(num)
    z_7s = torch.zeros(num)+z_11
    z_2s = torch.zeros(num)+z_22
    z_1s = torch.ones(num)
    z_0s = torch.zeros(num)


    cfPOTrain_t1z1 = potentialOutcomeSimulation(w,X,A,t_1s,beta0,w_beta_T2Y,bias_T2Y,epsiloncf,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_1s,proba=True )
    cfPOTrain_t1z0 = potentialOutcomeSimulation(w,X,A,t_1s,beta0,w_beta_T2Y,bias_T2Y,epsiloncf,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s,proba=True )
    cfPOTrain_t0z0 = potentialOutcomeSimulation(w,X,A,t_0s,beta0,w_beta_T2Y,bias_T2Y,epsiloncf,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_0s,proba=True )
    cfPOTrain_t0z7 = potentialOutcomeSimulation(w,X,A,t_0s,beta0,w_beta_T2Y,bias_T2Y,epsiloncf,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_7s,proba=True )
    cfPOTrain_t0z2 = potentialOutcomeSimulation(w,X,A,t_0s,beta0,w_beta_T2Y,bias_T2Y,epsiloncf,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=z_2s,proba=True )


    if gen_type == "train":

        my_data = {'T':np.array(T),
            'cfT':np.array(cfT_train),
            'features': np.array(X), 
            'PO':np.array(PO),
            'cfPO':np.array(cfPOTrain),
            'nodesToFlip':nodesToFlipTrain,
            'network':A,
            "meanT":meanT,

            "train_t1z1":np.array(cfPOTrain_t1z1),
            "train_t1z0":np.array(cfPOTrain_t1z0),
            "train_t0z0":np.array(cfPOTrain_t0z0),
            "train_t0z7":np.array(cfPOTrain_t0z7),
            "train_t0z2":np.array(cfPOTrain_t0z2),
        }
    if gen_type == "val":
        my_data = {'T':np.array(T),
            'cfT':np.array(cfT_train),
            'features': np.array(X), 
            'PO':np.array(PO),
            'cfPO':np.array(cfPOTrain),
            'nodesToFlip':nodesToFlipTrain,
            'network':A,
            "meanT":meanT,

            "val_t1z1":np.array(cfPOTrain_t1z1),
            "val_t1z0":np.array(cfPOTrain_t1z0),
            "val_t0z0":np.array(cfPOTrain_t0z0),
            "val_t0z7":np.array(cfPOTrain_t0z7),
            "val_t0z2":np.array(cfPOTrain_t0z2),
        }
    if gen_type == "test":
        my_data = {'T':np.array(T),
            'cfT':np.array(cfT_train),
            'features': np.array(X), 
            'PO':np.array(PO),
            'cfPO':np.array(cfPOTrain),
            'nodesToFlip':nodesToFlipTrain,
            'network':A,
            "meanT":meanT,

            "test_t1z1":np.array(cfPOTrain_t1z1),
            "test_t1z0":np.array(cfPOTrain_t1z0),
            "test_t0z0":np.array(cfPOTrain_t0z0),
            "test_t0z7":np.array(cfPOTrain_t0z7),
            "test_t0z2":np.array(cfPOTrain_t0z2),
        }
    return my_data

def simulate_data(dataset,w_c,w, w_beta_T2Y,beta0, betaConfounding,betaNeighborConfounding, 
                  bias_T2Y,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, 
                  betaNoise, setting,nodes = 2000,edges_new_node = 2, covariate_dim = 10,flipRate = 1,watts_strogatz = False):
    train =  generate_data(dataset = dataset,w_c = w_c, w= w, w_beta_T2Y=w_beta_T2Y,beta0=beta0, betaConfounding=betaConfounding,
                           betaNeighborConfounding=betaNeighborConfounding,bias_T2Y=bias_T2Y,
                           betaTreat2Outcome=betaTreat2Outcome,betaCovariate2Outcome=betaCovariate2Outcome,
                           betaNeighborCovariate2Outcome=betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome=betaNeighborTreatment2Outcome, 
                           betaNoise=betaNoise,gen_type = "train",nodes = nodes,edges_new_node = edges_new_node, covariate_dim = covariate_dim,
                           flipRate = flipRate,watts_strogatz=watts_strogatz)
    val = generate_data(dataset,w_c = w_c, w=w,w_beta_T2Y=w_beta_T2Y,beta0=beta0, betaConfounding=betaConfounding,
                            betaNeighborConfounding=betaNeighborConfounding,bias_T2Y=bias_T2Y,
                            betaTreat2Outcome=betaTreat2Outcome,betaCovariate2Outcome=betaCovariate2Outcome,
                            betaNeighborCovariate2Outcome=betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome=betaNeighborTreatment2Outcome, 
                            betaNoise=betaNoise,gen_type = "val",nodes = nodes,edges_new_node = edges_new_node, covariate_dim = covariate_dim,
                            flipRate = flipRate,watts_strogatz=watts_strogatz)
    test = generate_data(dataset,w_c= w_c,w=w,w_beta_T2Y=w_beta_T2Y,beta0=beta0, betaConfounding=betaConfounding,
                            betaNeighborConfounding=betaNeighborConfounding,bias_T2Y=bias_T2Y,
                            betaTreat2Outcome=betaTreat2Outcome,betaCovariate2Outcome=betaCovariate2Outcome,
                            betaNeighborCovariate2Outcome=betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome=betaNeighborTreatment2Outcome, 
                            betaNoise=betaNoise,gen_type = "test",nodes = nodes,edges_new_node = edges_new_node, covariate_dim = covariate_dim,
                            flipRate = flipRate,watts_strogatz=watts_strogatz)
    data = {"train":train,"val":val,"test":test}

   
    

    file = "data/simulated/" + setting +".pkl"

    with open(file,"wb") as f:
        pkl.dump(data,f)    
    

