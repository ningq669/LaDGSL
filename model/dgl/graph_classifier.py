from .rgcn_model import RGCN
from .LawareGAT import LaGAT
from dgl import mean_nodes
import torch.nn as nn
import torch
import numpy as np
import copy


class GraphClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.dropout = nn.Dropout(p=params.dropout)
        self.relu = nn.ReLU()
        self.train_rels = params.train_rels
        self.relations = params.num_rels
        self.init_emb = params.init_emb_dim
        self.hid_emb = params.hidden_dim
        self.pos_emb = params.inp_dim
        self.om_dim = params.feat_dim
        self.layer = params.num_gcn_layers
        self.device = params.device

        self.gnn = RGCN(params)
        self.gat = LaGAT(params)
        self.mp_layer1 = nn.Linear(self.om_dim, 256)
        self.mp_layer2 = nn.Linear(256, self.hid_emb)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_layer = nn.Linear(3 * (1 + self.layer) * (2 * self.hid_emb) + 2 * self.hid_emb, 512)
        self.fc_layer_1 = nn.Linear(512, 128)
        self.fc_layer_2 = nn.Linear(128, 1)

    def omics_feat(self, emb):
        self.genefeat = emb

    # Load multi-omics data.
    def get_omics_features(self, ids):
        a = []
        for i in ids:
            a.append(self.genefeat[i.cpu().numpy().item()])
        return np.array(a)

    def forward(self, data):

        # Dual GNN attention propagation layer based on link-aware attention.

        g = data
        g1 = copy.deepcopy(g)
        g2 = copy.deepcopy(g)

        g1.ndata['h'] = self.gnn(g1)
        g1_out = mean_nodes(g1, 'repr')

        g2.ndata['h_all'] = self.gat(g2)
        g2_out = mean_nodes(g2, 'h_all')

        g_out = torch.cat([g1_out, g2_out], dim=2)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs1 = g1.ndata['repr'][head_ids]
        head_embs2 = g2.ndata['h1'][head_ids]
        head_embs = torch.cat([head_embs1, head_embs2], dim=2)

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs1 = g1.ndata['repr'][tail_ids]
        tail_embs2 = g2.ndata['h2'][tail_ids]
        tail_embs = torch.cat([tail_embs1, tail_embs2], dim=2)

        head_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][head_ids])).to(self.device)
        tail_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][tail_ids])).to(self.device)

        if self.params.add_feat_emb:
            fuse_feat1 = self.mp_layer2(self.bn1(self.relu(self.mp_layer1(head_feat))))
            fuse_feat2 = self.mp_layer2(self.bn1(self.relu(self.mp_layer1(tail_feat))))
            fuse_feat = torch.cat([fuse_feat1, fuse_feat2], dim=1)

        # Fuse latent features with explicit features.
        g_rep = torch.cat(
            [g_out.view(-1, (1 + self.layer) * (self.hid_emb * 2)),
             head_embs.view(-1, (1 + self.layer) * (self.hid_emb * 2)),
             tail_embs.view(-1, (1 + self.layer) * (self.hid_emb * 2)),
             fuse_feat.view(-1, 2 * self.hid_emb)
             ], dim=1)

        output = self.fc_layer_2(self.relu(self.fc_layer_1(self.relu(self.fc_layer(self.dropout(g_rep))))))

        return (output, g_rep)
