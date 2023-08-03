import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl
from dgl.nn.pytorch import GraphConv
import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon import nn as ng


from layers import MultiHeadGATLayer, HAN_metapath_specific



class PNAGAT(nn.Module):
    def __init__(self, G0, meta_paths_list, feature_attn_size, num_heads, num_diseases, num_mirnas, num_lncrnas,
                 d_sim_dim, m_sim_dim, l_sim_dim, out_dim, dropout, slope, g_ml, g_dl, feat_ml, feat_dl):
        super(PNAGAT, self).__init__()

        self.G0 = G0
        self.meta_paths = meta_paths_list
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.num_lncrnas = num_lncrnas
        # self.GraphConv = GraphConv

        self.v1 = nn.Parameter(torch.ones(2, 1) * 0.5)
        self.v2 = nn.Parameter(torch.ones(2, 1) * 0.5)
        self.v3 = nn.Parameter(torch.ones(2, 1) * 0.5)
        self.v4 = nn.Parameter(torch.ones(2, 1) * 0.5)


        self.feat_ml = feat_ml
        self.g_ml = g_ml
        self.convm = dgl.nn.PNAConv(559, 512,
                              # aggregators=['mean', 'max', 'sum', 'var', 'moment3', 'moment4'],
                              # aggregators=['mean', 'max', 'sum', 'var', 'std', 'moment3', 'moment4'],
                              aggregators=['mean', 'moment4'],
                              scalers=['identity'],
                              delta=2.5)

        self.feat_dl = feat_dl
        self.g_dl = g_dl
        self.convd = dgl.nn.PNAConv(447, 512,
                              # aggregators=['mean', 'max', 'sum', 'var', 'moment3', 'moment4'],
                              # aggregators=['mean', 'max', 'sum', 'var', 'std', 'moment3', 'moment4'],
                              aggregators=['mean', 'moment4'],
                              scalers=['identity'],
                              delta=2.5)





        self.gat = MultiHeadGATLayer(G0, feature_attn_size, num_heads, dropout, slope)
        self.heads = nn.ModuleList()

        self.metapath_layers = nn.ModuleList()
        for i in range(self.num_heads):
            self.metapath_layers.append(HAN_metapath_specific(G0, feature_attn_size, out_dim, dropout, slope))

        self.dropout = nn.Dropout(dropout)

        self.d_fc0 = nn.Linear(383, out_dim)
        self.m_fc0 = nn.Linear(495, out_dim)
        # self.d_fc = nn.Linear(512, out_dim*2)
        # self.m_fc = nn.Linear(512, out_dim*2)


        self.h_fc = nn.Linear(out_dim, 256)
        # self.dropout = nn.Dropout(dropout)
        # self.m_fc = nn.Linear(m_sim_dim, out_dim, bias=False)
        # self.d_fc = nn.Linear(d_sim_dim, out_dim, bias=False)
        # self.fusion = nn.Linear(out_dim + feature_attn_size * num_heads, out_dim)
        self.predict = nn.Linear(out_dim, 1)
        # self.BilinearDecoder = BilinearDecoder(feature_size=64)
        # self.InnerProductDecoder = InnerProductDecoder()

    def forward(self, G0, G, diseases, mirnas):   # 这里试一试只用G训练集，不用到G0所有的数据

        index1 = 0
        for meta_path in self.meta_paths:

            if meta_path == 'ml':
                h_agg1 = self.convm(self.g_ml, self.feat_ml)  # 1345*512 tensor
                h_agg2 = self.gat(self.g_ml)

            elif meta_path == 'dl':
                h_agg3 = self.convd(self.g_dl, self.feat_dl)  # 1345*512 tensor
                h_agg4 = self.gat(self.g_dl)

        # disease0 = h_agg0[:383]
        # mirna0 = h_agg0[383:878]

        mirna1 = h_agg1[383:878]
        mirna2 = h_agg2[383:878]

        disease1 = h_agg3[:383]
        disease2 = h_agg4[:383]



        h1 = self.v1[0] * disease1 + self.v1[1] * disease2
        h2 = self.v2[0] * mirna1 + self.v2[0] * mirna2


        G_d_sim = G.ndata['d_sim'][:self.num_diseases]
        d_sim_f = self.d_fc0(G_d_sim)
        h_d = self.v3[0] * h1 + self.v3[1] * d_sim_f


        G_m_sim = G.ndata['m_sim'][self.num_diseases:878]
        m_sim_f = self.m_fc0(G_m_sim)
        h_m =self.v4[0] * h2 + self.v4[1] * m_sim_f



        # Decoder过程

        # h_d = self.dropout(F.elu(self.d_fc(h_d)))
        # h_m = self.dropout(F.elu(self.m_fc(h_m)))

        h = torch.cat((h_d, h_m), dim=0)    # （878,64）
        h = self.dropout(F.elu(self.h_fc(h)))

        h_diseases = h[diseases]    # disease中有重复的疾病名称;(17376,64)
        h_mirnas = h[mirnas]        # (17376,64)

        h_concat = torch.cat((h_diseases, h_mirnas), 1)         # (17376,128)
        predict_score = torch.sigmoid(self.predict(h_concat))   # (17376,128)->(17376,128*2)->(17376,1)

        # predict_score = self.BilinearDecoder(h_diseases, h_mirnas)
        # predict_score = self.InnerProductDecoder(h_diseases, h_mirnas)
        # predict_score = torch.unsqueeze(predict_score, 1)

        return predict_score






