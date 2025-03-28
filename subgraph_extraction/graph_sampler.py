import struct
import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.sparse as ssp
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius


def links2subgraphs(A, graphs, params, max_label_value=None):

    # Extract subgraphs.
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pairs'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += len(split['pairs']) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env, split_name):
        flag = 0
        pairs = []
        # Extract the node subgraph.
        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (idx, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                # Check the subgraph nodes numbers.
                nodes = datum['nodes']
                if len(nodes):
                    max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                    subgraph_sizes.append(datum['subgraph_size'])
                    enc_ratios.append(datum['enc_ratio'])
                    num_pruned_nodes.append(datum['num_pruned_nodes'])
                    str_id = '{:08}'.format(flag).encode('ascii')
                    # Save pairs.
                    pair = nodes[0:2] + [int(datum['r_label'])]
                    pairs.append(pair)
                    # Write the data.
                    with env.begin(write=True, db=split_env) as txn:
                        txn.put(str_id, serialize(datum))
                    flag += 1

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (flag).to_bytes(int.bit_length(len(links)), byteorder='little'))
        return pairs

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for gene pairs in {split_name} set")

        labels = np.ones(len(split['pairs']))
        db_name_pairs = split_name + '_pairs'
        split_env = env.open_db(db_name_pairs.encode())
        extraction_helper(A, split['pairs'], labels, split_env, split_name)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(),
                (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(),
                (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    lst = np.random.choice(len(links), sample_size)
    for idx in lst:
        (n1, n2, r_label) = links[idx]
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A,
                                                                                                   params.hop,
                                                                                                   params.enclosing_sub_graph,
                                                                                                   params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size,
                 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_,
                                                                                               params_.hop,
                                                                                               params_.enclosing_sub_graph,
                                                                                               params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels,
             'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None,
                                 max_node_label_value=None):
    # Extract the h-hop enclosing subgraphs around link 'ind'.
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T
    ind = list(ind)
    ind[0], ind[1] = int(ind[0]), int(ind[1])
    ind = (ind[0], ind[1])
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph.
    if enclosing_sub_graph:
        if ind[0] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        if ind[0] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):
    # Implementation of the node labeling scheme described in the paper.
    # Add position information.
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [
        np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for
        r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes
