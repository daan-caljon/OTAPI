import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.causal_models.layers import GraphConvolution

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

class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nclass)#.cuda()
        #self.dropout =dropout 
        self.dropout = nn.Dropout(dropout)#.cuda()
    

    def forward(self, x, adj):
        num = adj.shape[0]
        diag = torch.diag(torch.FloatTensor([1 for _ in range(num)])).to(device=x.device) #(torch.cuda.FloatTensor([1 for _ in range(num)]))
        x = F.relu(self.gc1(x, adj+diag))
        #x = F.dropout(x, self.dropout)
        x= self.dropout(x)
        return x

class NN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(NN,self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)#.cuda()
        #self.fc2 = nn.Linear(out_dim, out_dim).cuda() #fc2 is zelf toegevoegd
        self.act = nn.LeakyReLU(0.2, inplace=True)#.cuda()
        self.dropout = nn.Dropout(0.1)#.cuda()
        #self.sigmoid = nn.Sigmoid().cuda()
    def forward(self,x):
        #x = F.relu(self.fc(x)) --> change back maybe
        x = self.fc(x)
        x = self.act(x)
        #x= self.fc2(x)
        x = self.dropout(x)
        #x= self.sigmoid(x) #--> this might not work (added it myself)
        return x


class Predictor(nn.Module):

    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Predictor, self).__init__()

        self.predict1 = nn.Linear(input_size,hidden_size1)#.cuda()
        self.predict2 = nn.Linear(hidden_size1,hidden_size2)#.cuda()
        self.predict3 = nn.Linear(hidden_size2,output_size)#.cuda()
        self.act = nn.LeakyReLU(0.2, inplace=True)#.cuda()
        self.dropout = nn.Dropout(0.1)#.cuda()
        self.sigmoid = nn.Sigmoid()#.cuda()

    def forward(self,x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict3(x)
        x = self.sigmoid(x)
        return  x



class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Discriminator,self).__init__()

        self.disc1 = nn.Linear(input_size,hidden_size1)#.cuda()
        self.disc2 = nn.Linear(hidden_size1,hidden_size2)#.cuda()
        self.disc3 = nn.Linear(hidden_size2,output_size)#.cuda()
        self.act = nn.LeakyReLU(0.2, inplace=True)#.cuda()
        self.dropout = nn.Dropout(0.1)#.cuda().cuda()
        self.sigmoid = nn.Sigmoid()#.cuda()


    def forward(self,x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x
