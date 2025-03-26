import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .aggregators import SumAggregator, MLPAggregator

class Attention_light(nn.Module):
    def __init__(self, inp_dim, emb_dim):
        super(Attention_light, self).__init__()
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.out_features = emb_dim
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, query, nodefea, input_adj):
        # Calculating attention weights.
        e_fea = torch.mul(query, nodefea.clone())
        cor_node = torch.abs(torch.sum(e_fea, dim=-1, keepdim=True))
        wei_fea = torch.mul(cor_node, nodefea.clone())
        num_nei = input_adj.sum(dim=-1, keepdim=True)
        fea_up = self.leakyrelu(torch.matmul(input_adj, wei_fea))
        node_fea_norm = fea_up / (num_nei + 1e-8)
        out_feat_norm = self.layer_norm(node_fea_norm)
        return out_feat_norm


class Aggregator(nn.Module):
    def __init__(self, input_dim, activation: str = 'relu', initializer='normal'):
        super(Aggregator, self).__init__()

        self.activation = F.relu if activation == 'relu' else torch.tanh
        self.initializer = initializer

        self.w = nn.Parameter(torch.empty(input_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

        if self.initializer == 'normal':
            nn.init.normal_(self.w, mean=0.0, std=0.01)
        else:
            nn.init.normal_(self.w)

    def forward(self, entity, neighbor):

        combined = entity + neighbor
        out = torch.matmul(combined, self.w) + self.b

        return self.activation(out)


class LaGAT(nn.Module):
    def __init__(self, params):

        super(LaGAT, self).__init__()

        self.init_emb = params.init_emb_dim
        self.hid_emb = params.hidden_dim
        self.pos_emb = params.inp_dim
        self.om_dim = params.feat_dim
        self.layer = params.num_gcn_layers
        self.device = params.device

        self.num_nodes = params.num_nodes
        self.device = params.device
        self.add_transe_emb = params.add_transe_emb
        self.Att_light_one = Attention_light(self.hid_emb, self.hid_emb)
        self.Att_light = Attention_light(self.hid_emb, self.hid_emb)
        self.sumAgg_one = Aggregator(self.hid_emb)
        self.sumAgg = Aggregator(self.hid_emb)

        self.transform = nn.Parameter(torch.Tensor(self.init_emb + self.pos_emb, self.hid_emb), requires_grad=True)
        nn.init.xavier_uniform_(self.transform,
                                gain=nn.init.calculate_gain('relu'))

        if params.use_kge_embeddings:
            kg_embed = np.load('data/SynLethKG/kg_embedding/kg_TransE_l2_entity.npy')
            self.embed = torch.FloatTensor(kg_embed).to(params.device)
        else:
            self.embed = nn.Parameter(torch.Tensor(self.num_nodes, self.init_emb), requires_grad=True)
            nn.init.xavier_uniform_(self.embed,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, ano_g):

        batch_num_nodes = torch.tensor(ano_g.batch_num_nodes(), device=self.device)

        graph_indices = torch.repeat_interleave(torch.arange(len(batch_num_nodes), device=self.device), batch_num_nodes)

        adj_half = ((ano_g.adjacency_matrix()).to_dense()).to(self.device)
        adj = adj_half + adj_half.T
        adj[adj > 0] = 1

        N = adj.size(0)

        # Get initial features.
        init_feat = torch.cat([ano_g.ndata['feat'], self.embed[ano_g.ndata['idx']]], dim=1)
        feat = torch.matmul(init_feat, self.transform)

        head_ids = (ano_g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (ano_g.ndata['id'] == 2).nonzero().squeeze(1)

        batchsize = head_ids.size(0)

        all_head_ind = head_ids.index_select(0, graph_indices)
        all_tail_ind = tail_ids.index_select(0, graph_indices)

        all_c1_featQ = feat.index_select(0, all_head_ind)
        all_c2_featQ = feat.index_select(0, all_tail_ind)

        all_adj_I = torch.eye(adj.size(0)).to(self.device)
        all_center1_adj = all_adj_I[head_ids]
        all_center2_adj = all_adj_I[tail_ids]
        all_adj_powers = [all_adj_I]
        all_c1nei_list, all_c2nei_list = [all_center1_adj], [all_center2_adj]

        all_feat_head = feat.clone()
        all_feat_tail = feat.clone()

        # Get initial features.
        for depth in range(self.layer - 1):
            all_adj_k = torch.mm(all_adj_powers[depth], adj)

            all_adj_k.fill_diagonal_(1)
            all_adj_k[all_adj_k > 0] = 1

            all_adj_powers.append(all_adj_k)
            all_c1_nei = all_adj_k[head_ids]
            all_c2_nei = all_adj_k[tail_ids]

            all_c1nei_list.append(all_c1_nei)
            all_c2nei_list.append(all_c2_nei)

        index_head = all_c1nei_list[0]
        unq_index_head = torch.sum(index_head, dim=0)
        unq_index_head[unq_index_head > 0] = 1
        init_head = torch.mul(unq_index_head.unsqueeze(dim=-1), feat)
        index_tail = all_c2nei_list[0]
        unq_index_tail = torch.sum(index_tail, dim=0)
        unq_index_tail[unq_index_tail > 0] = 1
        init_tail = torch.mul(unq_index_tail.unsqueeze(dim=-1), feat)

        layer_add_all, layer_add_head, layer_add_tail = [feat], [init_head], [init_tail]

        # Link-aware GAT
        for hop in range(self.layer - 1, -1, -1):

            all_c1cur_adj = all_c1nei_list[hop]
            all_c2cur_adj = all_c2nei_list[hop]

            c1cur_adj_flat = torch.sum(all_c1cur_adj, dim=0)
            c2cur_adj_flat = torch.sum(all_c2cur_adj, dim=0)

            result_c1 = (c1cur_adj_flat >= 0) & (c1cur_adj_flat <= 2)
            result_c2 = (c2cur_adj_flat >= 0) & (c2cur_adj_flat <= 2)

            if not result_c1.all() or not result_c2.all():
                invalid_c1rows = torch.nonzero(~result_c1).squeeze()
                invalid_c2rows = torch.nonzero(~result_c2).squeeze()
                raise ValueError(
                    f"错误batch: c1cur_adj 无效行 {invalid_c1rows.tolist()}, c2cur_adj 无效行 {invalid_c2rows.tolist()}")

            all_mask_c1 = torch.mul(c1cur_adj_flat.unsqueeze(dim=-1), adj)
            all_mask_c2 = torch.mul(c2cur_adj_flat.unsqueeze(dim=-1), adj)

            all_c1nei_upemb = self.Att_light_one(all_c2_featQ, all_feat_head, all_mask_c1)
            all_c2nei_upemb = self.Att_light(all_c1_featQ, all_feat_tail, all_mask_c2)

            all_index_c1 = (c1cur_adj_flat.squeeze(dim=-1)).nonzero()
            c1_batch_node = all_index_c1[:, 0]
            c1_mask_feat = torch.mul(c1cur_adj_flat.unsqueeze(dim=-1), feat)
            Agg_c1 = self.sumAgg_one(c1_mask_feat, all_c1nei_upemb)
            all_feat_head[c1_batch_node, :] = Agg_c1[c1_batch_node, :]

            all_index_c2 = (c2cur_adj_flat.squeeze(dim=-1)).nonzero()
            c2_batch_node = all_index_c2[:, 0]
            c2_mask_feat = torch.mul(c2cur_adj_flat.unsqueeze(dim=-1), feat)
            Agg_c2 = self.sumAgg(c2_mask_feat, all_c2nei_upemb)
            all_feat_tail[c2_batch_node, :] = Agg_c2[c2_batch_node, :]

            all_feat_update = all_feat_head + all_feat_tail
            layer_add_all.append(all_feat_update)
            layer_add_head.append(all_feat_head)
            layer_add_tail.append(all_feat_tail)

        final_all_feat = (torch.cat(layer_add_all, dim=1)).view(N, self.layer + 1, -1)
        final_head_feat = (torch.cat(layer_add_head, dim=1)).view(N, self.layer + 1, -1)
        final_tail_feat = (torch.cat(layer_add_tail, dim=1)).view(N, self.layer + 1, -1)

        ano_g.ndata['h_all'] = final_all_feat
        ano_g.ndata['h1'] = final_head_feat
        ano_g.ndata['h2'] = final_tail_feat

        return ano_g.ndata['h_all']
