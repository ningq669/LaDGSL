import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, num_nodes=122343142, dropout=0.0,
                 edge_dropout=0.0, is_input_layer=False, embed=False, add_transe_emb=True):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.num_nodes = num_nodes
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))
        self.add_transe_emb = add_transe_emb
        self.aggregator = aggregator
        self.is_input_layer = is_input_layer
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()
        # if is_input_layer:
        if embed is not None:
            self.embed = embed

        self.transform = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.transform,
                                gain=nn.init.calculate_gain('relu'))

    # Define how propagation is done in subclass.
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, attn_rel_emb=None):

        # Calculate the structural characteristics of the graph.
        init_feat = torch.cat([g.ndata['feat'], self.embed[g.ndata['idx']]], dim=1)
        g.ndata['init_fea'] = torch.matmul(init_feat, self.transform)

        self.propagate(g, attn_rel_emb)

        # Apply bias and activation.
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        # Initial layer and hidden layer
        if self.is_input_layer and self.add_transe_emb:
            init = g.ndata['init_fea']
            x = torch.cat([init, g.ndata['h']], dim=1)
            g.ndata['repr'] = x.unsqueeze(1).reshape(-1, 2, self.out_dim)
        elif self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, num_nodes=12345342,
                 bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, embed=None,
                 add_transe_emb=True,
                 one_attn=False):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            num_nodes=num_nodes,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer,
            embed=embed,
            add_transe_emb=add_transe_emb)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.num_nodes = num_nodes
        self.add_transe_emb = add_transe_emb
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.out_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        self.one_attn = one_attn

        if self.has_attn:
            self.A = nn.Linear(2 * self.out_dim + self.attn_rel_emb_dim, self.out_dim)
            self.B = nn.Linear(self.out_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None, nonKG=True):

        weight = self.weight.view(self.num_bases, self.out_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.out_dim, self.out_dim)

        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))

        input_F = 'init_fea' if self.is_input_layer else 'h'

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])

            if input_F == 'init_fea' and self.add_transe_emb:
                x = edges.src[input_F]
                y = edges.dst[input_F]
            else:
                x = edges.src['feat']  # Position-only feature
                x = edges.dst['feat']

            msg = edges.data['w'] * torch.bmm(x.unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(y, self.self_loop_weight)  # batch_size*fea_size

            # Calculate attention.
            if self.has_attn:
                e = torch.cat([x, y, attn_rel_emb(edges.data['type'])], dim=1)

                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)

            edges.data['a'] = a

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)
