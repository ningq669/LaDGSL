import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape  # List of sparse adjacency matrices.

    flag = 0
    for adj in adj_list:
        if flag > 1:
            adjcoo = adj.tocoo()
            rows += adjcoo.row.tolist()
            cols += adjcoo.col.tolist()
            dats += adjcoo.data.tolist()
        flag += 1
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    # Converting ssp multigraph to dgl multigraph.

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):

        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # Make dgl graph.
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # Add node features.
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


# Merge multiple graph samples.
def collate_dgl(samples):
    graphs_pos, g_labels_pos, r_labels_pos = map(list, zip(*samples))

    batched_graph_pos = dgl.batch(graphs_pos)

    return (batched_graph_pos, r_labels_pos), g_labels_pos


# The subgraph is released to the unified device.
def move_batch_to_device_dgl(batch, device):
    (g_dgl_pos, r_labels_pos), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.FloatTensor(r_labels_pos).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)

    return g_dgl_pos, r_labels_pos, targets_pos


def send_graph_to_device(g, device):
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g


# Calculate the structural characteristics of the graph.
def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())
