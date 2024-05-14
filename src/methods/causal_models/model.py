import argparse
import torch
import pickle as pkl
import torch.nn as nn
import utils.utils as utils
import numpy as np
from src.methods.causal_models.modules import GCN, NN, Predictor,Discriminator
import torch.nn as nn
import torch.nn.functional as F
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
class NetEstimator(nn.Module):

    def __init__(self,Xshape,hidden,dropout):
        super(NetEstimator, self).__init__()
        self.encodergc = GCN(nfeat=Xshape, nclass=Xshape, dropout=dropout) #change back to nclass = Xshape, self.encodergc
        self.encoder = NN(in_dim=Xshape+Xshape, out_dim=hidden)
        #self.encoder_final = NN(in_dim=hidden, out_dim=hidden)
        self.predictor = Predictor(input_size=hidden + 2, hidden_size1=hidden, hidden_size2=hidden,output_size=1)
        self.discriminator = Discriminator(input_size=hidden,hidden_size1=hidden,hidden_size2=hidden,output_size=1)
        self.discriminator_z = Discriminator(input_size=hidden+1,hidden_size1=hidden,hidden_size2=hidden,output_size=1)
    

    def forward(self,A,X,T,Z=None):
        #This was changed slightly compared to the original paper/code
        embeddingsgc = self.encodergc(X,A)#self.encodergc(X, A) embeddingsgc
        embeddings = self.encoder(torch.cat((embeddingsgc, X), 1)) 
        #embeddings = self.encoder_final(embeddings)
        pred_treatment = self.discriminator(embeddings)
        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z
        embed_treatment = torch.cat((embeddings, T.reshape(-1, 1)), 1) 
        pred_z = self.discriminator_z(embed_treatment)
        embed_treatment_avgT = torch.cat((embed_treatment, neighborAverageT.reshape(-1, 1)), 1)
        pred_outcome0 = self.predictor(embed_treatment_avgT).view(-1)

        return pred_treatment,pred_z,pred_outcome0,embeddings, neighborAverageT