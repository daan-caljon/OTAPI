import argparse
import torch
import time
import src.utils.utils as utils
import numpy as np
from src.methods.causal_models.model import NetEstimator
from src.methods.causal_models.baselineModels import GCN_DECONF,CFR,GCN_DECONF_INTERFERENCE,CFR_INTERFERENCE
from src.methods.causal_models.model_training import Experiment
import itertools

def run_model(dataset,model,epochs,lr,setting, n_in = 1, n_out = 1):
    """
    This is code from Song Jiang: https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
    This code was ued in "Estimating causal effects on networked observational data":
    @inproceedings{netest2022,
    title={Estimating Causal Effects on Networked Observational Data via Representation Learning},
    author={Song Jiang, Yizhou Sun},
    booktitle={Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
    year={2022}
    }
    MIT License

    Copyright (c) 2022 Song Jiang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    #To change as little from the original code as possible, we keep parser but use the default values

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=7,help='Use CUDA training.')
    parser.add_argument('--seed', type=int, default=24, help='Random seed. RIP KOBE')
    parser.add_argument('--dataset', type=str, default='BC')#["BC","Flickr","full_sim"]
    parser.add_argument('--expID', type=int, default=4)
    parser.add_argument('--flipRate', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5,help='trade-off of p(t|x).')
    parser.add_argument('--gamma', type=float, default=0.5,help='trade-off of p(z|x,t).')
    parser.add_argument('--epochs', type=int, default=300,help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate.')
    parser.add_argument('--lrD', type=float, default=1e-3,help='Initial learning rate of Discriminator.')
    parser.add_argument('--lrD_z', type=float, default=1e-3,help='Initial learning rate of Discriminator_z.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dstep', type=int, default=50,help='epoch of training discriminator')
    parser.add_argument('--d_zstep', type=int, default=50,help='epoch of training discriminator_z')
    parser.add_argument('--pstep', type=int, default=1,help='epoch of training')
    parser.add_argument('--normy', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16,help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1,help='Dropout rate (1 - keep probability).')
    parser.add_argument('--save_intermediate', type=int, default=1,help='Save training curve and imtermediate embeddings')
    parser.add_argument('--model', type=str, default='NetEstimator',help='Models or baselines')
    #["NetEstimator","ND","TARNet","TARNet_INTERFERENCE","CFR","ND_INTERFERENCE","CFR_INTERFERENCE"]
    parser.add_argument('--alpha_base', type=float, default=0.5,help='trade-off of balance for baselines.')
    parser.add_argument('--printDisc', type=int, default=0,help='Print discriminator result for debug usage')
    parser.add_argument('--printDisc_z', type=int, default=0,help='Print discriminator_z result for debug usage')
    parser.add_argument('--printPred', type=int, default=1,help='Print encoder-predictor result for debug usage')


    startTime = time.time()

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.epochs = epochs
    args.lr = lr
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dataset = dataset
    args.model = model
    args.hidden = 16
    args.alpha_base = 0

    trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT,POVal,cfPOVal,testA,testX, testT,cfTestT,POTest,cfPOTest,\
        train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2 = utils.load_data(dataset,setting)

    print("TrainA",trainA, trainA.shape)
    print("trainX",trainX, trainX.shape)
    print("trainT",trainT, trainT.shape)
    print("POtrain",POTrain, POTrain.shape)
    print("cfPOTrain",cfPOTrain, cfPOTrain.shape)



    if args.model == "NetEstimator":
        model = NetEstimator(Xshape=trainX.shape[1],hidden=args.hidden,dropout=args.dropout)
    elif args.model == "ND":
        model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
    elif args.model == "TARNet":
        model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
    elif args.model == "CFR":
        model = CFR(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout,n_in=n_in,n_out=n_out)

    elif args.model == "CFR_INTERFERENCE":
        model = CFR_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
    elif args.model == "ND_INTERFERENCE":
        model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
    elif args.model == "TARNet_INTERFERENCE":
        model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)



    exp = Experiment(args,model,trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT,POVal,cfPOVal,testA, testX, testT,cfTestT,POTest,cfPOTest,\
        train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2,setting)

    """Train the model"""
    exp.train()

    """Moel Predicting"""
    val_loss = exp.predict()

    #save the model

    exp.save_model()


    print("Time usage:{:.4f} mins".format((time.time()-startTime)/60))
    print ("================================Setting again================================")
    print ("Model:{} Dataset:{}, expID:{}, filpRate:{}, alpha:{}, gamma:{}".format(args.model,args.dataset,args.expID,args.flipRate,args.alpha,args.gamma))
    print ("================================BYE================================")
    return val_loss

def train_best_models(dataset,setting,no_netest=False,no_CFR = False):
    """
    This function is used to tune the TARNet/CFR and Netest models
    The best model is saved
    :param dataset: str: name of the dataset
    :param setting: str: setting of the experiment
    :param no_netest: bool: if True, the Netest model is not trained
    :param no_CFR: bool: if True, the CFR model is not trained
    :return: dict: best_config_netest: best configuration for Netest
    :return: float: best_val_loss_netest: validation loss for Netest
    :return: dict: best_config_CFR: best configuration for CFR
    :return: float: best_val_loss_CFR: validation loss for CFR
    """
    hyperparameters_netest= {"epochs":[200,500,800], #200,500,1000
                  "lr":[5e-3,1e-3,5e-4]}  #5e-3,1e-3,1e-4
    hyperparameters_CFR = {"epochs":[500,800], #500,1000
                  "lr":[5e-3,1e-3,5e-4],
                  "n_in": [1,2],
                  "n_out":[1,2]} #5e-3,1e-3,1e-4

    keys = list(hyperparameters_netest.keys())
    values = list(hyperparameters_netest.values())

    # Compute the Cartesian product
    combinations = list(itertools.product(*values))

    # Generate dictionaries for each combination
    dict_combinations = [dict(zip(keys, combo)) for combo in combinations]

    # train the model and use the best config (in terms of validation loss)
    best_config_netest = None
    best_val_loss_netest = 100000


    for config in dict_combinations:
        epochs = config["epochs"]
        lr = config["lr"]
        val_loss = run_model(dataset,"NetEstimator",epochs,lr,setting)
        if val_loss < best_val_loss_netest:
            best_val_loss_netest = val_loss
            best_config_netest = config

    keys = list(hyperparameters_CFR.keys())
    values = list(hyperparameters_CFR.values())

    # Compute combinations
    combinations = list(itertools.product(*values))
    
    # Generate dictionaries for each combination
    dict_combinations = [dict(zip(keys, combo)) for combo in combinations]

    #
    best_val_loss_CFR = 100000
    best_config_CFR = None
    

    for config in dict_combinations:
        epochs = config["epochs"]
        lr = config["lr"]
        n_in = config["n_in"]
        n_out = config["n_out"]
        
        val_loss = run_model(dataset,"CFR",epochs,lr,setting,n_in=n_in,n_out= n_out)
        if val_loss < best_val_loss_CFR:
            best_val_loss_CFR = val_loss
            best_config_CFR = config
    
    # save models with best config
    
    run_model(dataset,"NetEstimator",best_config_netest["epochs"],best_config_netest["lr"],setting)
    
    run_model(dataset,"CFR",best_config_CFR["epochs"],best_config_CFR["lr"],setting)
    return best_config_netest,best_val_loss_netest,best_config_CFR,best_val_loss_CFR
    
    