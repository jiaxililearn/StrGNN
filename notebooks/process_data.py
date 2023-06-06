# %%
import numpy as np
import scipy.sparse as ssp
from tqdm import tqdm
import random
from tempfile import TemporaryFile

# # DGraph Preprocessing

# %%
window_size = 5

# %%
dgraph_data = np.load(
    # "/Users/jl102430/Documents/study/anomaly_detection/data/dynamic/DGraph/DGraphFin/dgraphfin.npz"
    "../../HRGCN/dataset/raw_data/dgraphfin.npz"
)

X = dgraph_data["x"]
y = dgraph_data["y"]

edge_index = dgraph_data["edge_index"]
edge_type = dgraph_data["edge_type"]
edge_timestamp = dgraph_data["edge_timestamp"]

train_mask = dgraph_data["train_mask"]
valid_mask = dgraph_data["valid_mask"]
test_mask = dgraph_data["test_mask"]

print(
    f"""
X shape: {X.shape},
y shape: {y.shape}

edge_index shape: {edge_index.shape}
edge_type shape: {edge_type.shape}
edge_timestamp shape: {edge_timestamp.shape}

train_mask shape: {train_mask.shape}
valid_mask shape: {valid_mask.shape}
test_mask shape: {test_mask.shape}
"""
)


# %%
def reindex_graph_dataset(
    edge_index, node_feature, node_label, edge_type, edge_timestamp, mask
):
    masked_edge_index = edge_index[mask]
    masked_edge_type = edge_type[mask]
    masked_edge_timestamp = edge_timestamp[mask] - 1

    sorted_index = np.argsort(masked_edge_timestamp)
    masked_edge_index = masked_edge_index[sorted_index]
    masked_edge_type = masked_edge_type[sorted_index]
    masked_edge_timestamp = masked_edge_timestamp[sorted_index]

    node_list = np.unique(masked_edge_index.flatten())

    reindex_edge_index = np.empty_like(masked_edge_index)
    node2id = {n: i for i, n in enumerate(node_list)}

    reindex_edge_index[:, 0] = np.array(
        list(map(lambda x: node2id[x], masked_edge_index[:, 0]))
    )
    reindex_edge_index[:, 1] = np.array(
        list(map(lambda x: node2id[x], masked_edge_index[:, 1]))
    )

    masked_node_feature = node_feature[node_list]
    masked_node_label = node_label[node_list]

    net = []
    for ts in tqdm(np.unique(masked_edge_timestamp)):
        ts_mask = masked_edge_timestamp == ts
        net_edge_index = reindex_edge_index[ts_mask]
        net_edge_type = masked_edge_type[ts_mask]

        _net = ssp.csc_matrix(
            (
                np.ones(net_edge_index.shape[0]),
                (net_edge_index[:, 0], net_edge_index[:, 1]),
            ),
            shape=(masked_node_feature.shape[0], masked_node_feature.shape[0]),
        )
        net.append(_net)
    net = np.array(net)

    print(
        f"""
          reindex_edge_index: {reindex_edge_index.shape}
          masked_edge_type: {masked_edge_type.shape}
          masked_edge_timestamp: {masked_edge_timestamp.shape}
          unique nodes: {np.unique(reindex_edge_index.flatten()).shape}
          masked_node_feature: {masked_node_feature.shape}
          masked_node_label: {masked_node_label.shape}
          net: {net.shape}
          """
    )

    return (
        reindex_edge_index,
        masked_node_feature,
        masked_node_label,
        masked_edge_type,
        masked_edge_timestamp,
        net,
        node2id,
    )


def sample_neg(net, pos):
    def sample_neg_ts(_net, pos):
        neg = []
        num_node = _net.shape[0]
        print(f"Sampling {pos.shape[0]} negative edges ..")
        pbar = tqdm(total=100)
        row, col = _net.nonzero()

        net_copy = ssp.lil_matrix(_net).copy()

        while len(neg) < row.shape[0]:
            i = random.randint(0, num_node - 1)
            j = random.randint(0, num_node - 1)

            if net_copy[i, j] == 0.0:
                neg.append([i, j])
                net_copy[i, j] = 1.0
                pbar.update(100 / pos.shape[0])
            # print(len(neg) / pos.shape[0])
        pbar.close()
        return neg

    negs = []
    for ts in tqdm(range(net.shape[0])):
        # print(f'Timestamp {ts}')
        negs.append(sample_neg_ts(net[ts], pos))

    return np.concatenate(negs, axis=1)


# %%
(
    train_edge_index,  # train_pos
    train_node_feature,
    train_node_label,
    train_edge_type,
    train_edge_timestamp,  # train_pos_id
    train_net,
    train_node2id,
) = reindex_graph_dataset(edge_index, X, y, edge_type, edge_timestamp, train_mask)

train_neg = sample_neg(train_net, train_edge_index)

# outfile = TemporaryFile()
np.savez(
    "./dgraph_train.npz",
    train_pos=train_edge_index,
    train_pos_id=train_edge_timestamp,
    train_neg=train_neg,
    train_neg_id=train_edge_timestamp,
)
print("Train .npz Saved!")

(
    valid_edge_index,  # valid_pos
    valid_node_feature,
    valid_node_label,
    valid_edge_type,
    valid_edge_timestamp,  # valid_pos_id
    valid_net,
    valid_node2id,
) = reindex_graph_dataset(edge_index, X, y, edge_type, edge_timestamp, valid_mask)

valid_neg = sample_neg(valid_net, valid_edge_index)
np.savez(
    "./dgraph_valid.npz",
    valid_pos=valid_edge_index,
    valid_pos_id=valid_edge_timestamp,
    valid_neg=valid_neg,
    valid_neg_id=valid_edge_timestamp,
)

print("Valid .npz Saved!")