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
class GCN_DECONF(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=True):
        super(GCN_DECONF, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid).cuda())
        else:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep,neighborAverageT



class CFR(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=2, n_out=2, cuda=True):
        super(CFR, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid,nhid).cuda())
        else:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self,edge_index, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)
        y = torch.where(t > 0,y1,y0)
        y = self.pp_act(y)
        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep,neighborAverageT




class GCN_DECONF_INTERFERENCE(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=True):
        super(GCN_DECONF_INTERFERENCE, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid).cuda())
        else:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = nn.ModuleList([nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = nn.ModuleList([nn.Linear(nhid+1,nhid) for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid+1,nhid) for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        rep1 = torch.cat((rep, neighborAverageT.reshape(-1, 1)), 1)
        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep1))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep1))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep1,neighborAverageT



class CFR_INTERFERENCE(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=True):
        super(CFR_INTERFERENCE, self).__init__()
        
        
        if cuda:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid,nhid).cuda())
        else:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = nn.ModuleList([nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout =dropout
        self.sigmoid = nn.Sigmoid().cuda()
        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            rep = F.dropout(rep, self.dropout, training=self.training)

        rep1 = torch.cat((rep, neighborAverageT.reshape(-1, 1)), 1)
        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep1))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep1))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y0 = self.sigmoid(y0)
        y1 = self.out_t11(y10).view(-1)
        y1 = self.sigmoid(y1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep1,neighborAverageT

