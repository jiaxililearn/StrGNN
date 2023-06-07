import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os
import os.path
import random
import argparse
import pickle
import networkx as nx
from tqdm import tqdm

sys.path.append("%s/../pytorch_DGCNN" % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from os import path


def load_dataset(name):
    net_path = f"../data/dgraph_{name}_net.npy"
    dataset_path = f"../data/dgraph_{name}.npz"

    net = np.load(net_path, allow_pickle=True)
    data = np.load(dataset_path)

    pos = data[f"{name}_pos"]
    pos_id = data[f"{name}_pos_id"]
    neg = data[f"{name}_neg"]
    neg_id = data[f"{name}_neg_id"]

    node_features = data[f"{name}_node_feature"]
    node_labels = data[f"{name}_node_label"]

    print(
        f"""
          {name}_pos: {pos.shape}
          {name}_pos_id: {pos_id.shape}
          {name}_neg: {neg.shape}
          {name}_neg_id: {neg_id.shape}
          {name}_node_features: {node_features.shape}
          {name}_node_labels: {node_labels.shape}
          {name}_net: {net.shape}
          """
    )

    return net, pos, neg, pos_id, neg_id, node_features, node_labels


def dyn_links2subgraphs_dgraph(
    net,
    window_size,
    pos_id,
    pos,
    neg,
    neg_id,
    h=1,
    max_nodes_per_hop=None,
    node_features=None,
    node_labels=None,
):
    def helper(net, pos, pos_id, g_label, window_size):
        g_list = []
        # for n in tqdm(np.unique(pos_id)):
        for n in tqdm([5]):
            d_list = []
            for g_id in tqdm(range(n - window_size + 1, n + 1)):
                # g, n_labels, n_features = subgraph_extraction_labeling(
                #     (i, j), net[g_id], h, max_nodes_per_hop, None
                # )
                #  TODO: direct produce g without subgrah sampling
                g = nx.from_scipy_sparse_matrix(net[g_id])
                d_list.append(GNNGraph(g, g_label, node_labels, node_features))
            g_list.append(d_list)

        return g_list

    graphs = helper(net, pos, pos_id, 1, window_size) + helper(
        net, neg, neg_id, 0, window_size
    )
    return graphs


if __name__ == "__main__":
    """ Configuration """
    is_gpu = torch.cuda.is_available()
    if is_gpu:
        mode = "gpu"
    else:
        mode = "cpu"
    conf = dict(
        # window_size=5,
        # learning_rate=1e-4,
        gpu=is_gpu,
        num_epochs=50,
        batch_size=32,
        # mode=mode,
    )

    cmd_args.gm = "DGCNN"
    cmd_args.sortpooling_k = 0.6
    cmd_args.latent_dim = [32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = mode
    cmd_args.num_epochs = 50
    cmd_args.learning_rate = 1e-4
    cmd_args.batch_size = 32
    cmd_args.printAUC = True
    cmd_args.feat_dim = 17
    cmd_args.attr_dim = 0
    cmd_args.window = 5
    """ End of Configuration """

    """ Prepare Datasets """
    (
        train_net,
        train_pos,
        train_neg,
        train_pos_id,
        train_neg_id,
        train_node_features,
        train_node_labels,
    ) = load_dataset("train")
    (
        valid_net,
        valid_pos,
        valid_neg,
        valid_pos_id,
        valid_neg_id,
        valid_node_features,
        valid_node_labels,
    ) = load_dataset("valid")

    train_graphs = dyn_links2subgraphs_dgraph(
        train_net,
        window_size=cmd_args.window,
        pos_id=train_pos_id,
        pos=train_pos,
        neg=train_neg,
        neg_id=train_neg_id,
        h=1,
        node_features=train_node_features,
        node_labels=train_node_labels,
    )
    print(f"Train Graphs: {len(train_graphs)}")
    print(f"{train_graphs[:5]}")

    valid_graphs = dyn_links2subgraphs_dgraph(
        valid_net,
        window_size=cmd_args.window,
        pos_id=valid_pos_id,
        pos=valid_pos,
        neg=valid_neg,
        neg_id=valid_neg_id,
        h=1,
        node_features=valid_node_features,
        node_labels=valid_node_labels,
    )
    print(f"Valid Graphs: {len(valid_graphs)}")
    print(f"{valid_graphs[:5]}")

    """ Pre Config """
    if cmd_args.sortpooling_k <= 1:
        A = []
        for i in train_graphs:
            # print(type(i[-1]))
            A.append(i[-1])
        for i in valid_graphs:
            A.append(i[-1])
        # print(type(A[0]))
        num_nodes_list = sorted([g.num_nodes for g in A])
        cmd_args.sortpooling_k = num_nodes_list[
            int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
        ]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print(("k used in SortPooling is: " + str(cmd_args.sortpooling_k)))

    """ Model Def and Train """
    classifier = Classifier()

    if conf["gpu"]:
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    best_auc = 0

    for epoch in range(conf["num_epochs"]):
        """Train"""
        random.shuffle(train_idxes)
        classifier.train()

        avg_loss = loop_dataset(
            train_graphs,
            classifier,
            train_idxes,
            optimizer=optimizer,
            bsize=conf["batch_size"],
        )
        break
    print(f"avg_loss: {avg_loss}")

    """ Eval"""
