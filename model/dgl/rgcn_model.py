import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RGCNBasisLayer as RGCNLayer
import numpy as np

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator


class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        self.max_label_value = params.max_label_value
        self.init_emb = params.init_emb_dim
        self.hid_emb = params.hidden_dim
        self.pos_emb = params.inp_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn
        self.num_nodes = params.num_nodes
        self.device = params.device
        self.add_transe_emb = params.add_transe_emb

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        if params.use_kge_embeddings:
            kg_embed = np.load('data/SynLethKG/kg_embedding/kg_TransE_l2_entity.npy')
            self.embed = torch.FloatTensor(kg_embed).to(params.device)
        else:
            self.embed = nn.Parameter(torch.Tensor(self.num_nodes, self.init_emb), requires_grad=True)
            nn.init.xavier_uniform_(self.embed,
                                    gain=nn.init.calculate_gain('relu'))

        # Initialize aggregators for input and hidden layers.
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.hid_emb)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.hid_emb)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.hid_emb)

        # Create rgcn layers.
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()

        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)

        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.init_emb + self.pos_emb,
                         self.hid_emb,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         embed=self.embed,
                         num_nodes=self.num_nodes,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn,
                         add_transe_emb=self.add_transe_emb,
                         one_attn=True)

    def build_hidden_layer(self, idx):
        return RGCNLayer(
            self.init_emb + self.pos_emb,
            self.hid_emb,
            self.aggregator,
            self.attn_rel_emb_dim,
            self.aug_num_rels,
            self.num_bases,
            embed=self.embed,
            activation=F.relu,
            dropout=self.dropout,
            edge_dropout=self.edge_dropout,
            has_attn=self.has_attn,
            add_transe_emb=self.add_transe_emb,
            one_attn=True)

    def forward(self, g):
        for layer in self.layers:
            layer(g, self.attn_rel_emb)
        return g.ndata.pop('h')
