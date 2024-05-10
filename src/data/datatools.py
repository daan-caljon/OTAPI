import numpy as np
import scipy.io as sio
import pickle as pkl
from sklearn.decomposition import LatentDirichletAllocation

"""
    This is code from Song Jiang: https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
    This code was ued in "Estimating causal effects on networked observational data":
    @inproceedings{netest2022,
    title={Estimating Causal Effects on Networked Observational Data via Representation Learning},
    author={Song Jiang, Yizhou Sun},
    booktitle={Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
    year={2022}
    }
"""

def readData(dataset):
    if dataset == "BC":
        data = sio.loadmat(r"data/semi_synthetic/BC/BC0.mat")
        with open(r'data/semi_synthetic/BC/BC_parts.pkl','rb') as f:
            parts = pkl.load(f)
    
    if dataset == "Flickr":
        print ("Flickr")
        data = sio.loadmat(r"data/semi_synthetic/Flickr/Flickr01.mat")
        with open(r'data/semi_synthetic/Flickr/Flickr_parts.pkl','rb') as f:
            parts = pkl.load(f)
            
    return data,parts


def saveData(dataset,data,expID,flipRate):
    if dataset == "BC":
        file = "../BC/simulation/"+str(dataset)+"_fliprate_"+str(flipRate)+"_expID_"+str(expID)+".pkl"
    if dataset == "Flickr":
        print ("Flickr Flickr Save")
        file = "../Flickr/simulation/"+str(dataset)+"_fliprate_"+str(flipRate)+"_expID_"+str(expID)+".pkl"
        
    with open(file,'wb') as f:
        pkl.dump(data,f)



def dataSplit(parts):
    trainIndex = []
    valIndex = []
    testIndex = []
    for i in range(len(parts["parts"])):
        if parts["parts"][i]==0:
            trainIndex.append(i)
        elif parts["parts"][i]==1:
            valIndex.append(i)
        else:
            testIndex.append(i)
    print ("Size of train graph:{}, val graph:{}, test graph:{}".format(len(trainIndex),len(valIndex),len(testIndex)))
    return trainIndex,valIndex,testIndex


def covariateTransform(data,dimension,trainIndex,valIndex,testIndex):

    X = data["Attributes"]
    print("features shape:{}".format(X.shape))
    lda = LatentDirichletAllocation(n_components=dimension)
    lda.fit(X)
    X = lda.transform(X)
    trainX = X[trainIndex]
    valX = X[valIndex]
    testX = X[testIndex]
    print ("Shape of graph covariate train:{}, val:{}, test:{}".format(trainX.shape,valX.shape,testX.shape))

    return trainX,valX,testX


def adjMatrixSplit(data,trainIndex,valIndex,testIndex,dataset):
    if dataset == "Flickr":
        A = data["Network"]
    else:
        A = data["Network"].toarray()

    trainA = np.array([a[trainIndex] for a in A[trainIndex]])
    valA = np.array([a[valIndex] for a in A[valIndex]])
    testA = np.array([a[testIndex] for a in A[testIndex]])
    print ("Shape of adj matrix train:{}, val:{}, test:{}".format(trainA.shape,valA.shape,testA.shape))
    print("Number of edges in train graph:{}, val graph:{}, test graph:{}".format(np.sum(trainA)/2,np.sum(valA)/2,np.sum(testA)/2))

    return trainA,valA,testA


def sigmod(x):
    return 1/(1 + np.exp(-x))


def treatmentSimulation(w_c,X,A,betaConfounding,betaNeighborConfounding):

    covariate2TreatmentMechanism = sigmod(np.matmul(w_c,X.T))
    neighbors = np.sum(A,1)
    print (np.sum(neighbors==0))
    neighborAverage = np.divide(np.matmul(A, covariate2TreatmentMechanism.reshape(-1)), neighbors)
    print (np.mean(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage),np.std(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage))
    propensityT= sigmod(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage)
    meanT = np.mean(propensityT)
    T = np.array([1 if x>meanT else 0 for x in propensityT])
    print ("Lenght of treatment vector:{}".format(len(T)))

    return T,meanT


def noiseSimulation(data,trainIndex,valIndex,testIndex):

    X = data["Attributes"]
    epsilon = np.random.normal(0,1,X.shape[0])
    epsilonTrain = epsilon[trainIndex]
    epsilonVal = epsilon[valIndex]
    epsilonTest = epsilon[testIndex]

    return epsilonTrain,epsilonVal,epsilonTest


def potentialOutcomeSimulation(w,X,A,T,epsilon,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise,Z=None):

    covariate2OutcomeMechanism = sigmod(np.matmul(w,X.T))
    neighbors = np.sum(A,1)
    neighborAverage = np.divide(np.matmul(A, covariate2OutcomeMechanism.reshape(-1)), neighbors)

    if Z is None:
        print ("generate Z")
        neighborAverageT = np.divide(np.matmul(A, T.reshape(-1)), neighbors)
    else:
        print ("use Z")
        neighborAverageT = Z
    potentialOutcome = betaTreat2Outcome*T + betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage+betaNeighborTreatment2Outcome*neighborAverageT+betaNoise*epsilon
    print ("Lenght of potentialOutcome vector:{}".format(len(potentialOutcome)))

    return potentialOutcome


def flipTreatment(T,rate):
    
    numToFlip = int(len(T)*rate)
    nodesToFlip = set(np.random.choice(len(T), numToFlip, replace=False))
    cfT = np.array([1-T[i] if i in nodesToFlip else T[i] for i in range(len(T))])
    
    return cfT,nodesToFlip
    

