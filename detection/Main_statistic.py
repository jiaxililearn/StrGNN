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
sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from os import path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Anomaly detection')
# general settings
parser.add_argument('--data-name', default='USAir', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.815,
                    help='ratio of test links')
parser.add_argument('--window', type=int, default=5,
                    help='window size')
parser.add_argument('--graph', default='acc_digg.npy')
parser.add_argument('--split', default='digg0.1')
parser.add_argument('--gpu', default='1', help='gpu number')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))

    net = np.load('data_sta/'+args.graph)

    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    #Sample train and test links
    f = np.load('data_sta/'+args.split+'.npz')
    train_pos_id, train_neg_id, test_pos_id, test_neg_id = f['train_pos_id'], f['train_neg_id'], f['test_pos_id'], f['test_neg_id']
    train_pos, train_neg, test_pos, test_neg = f['train_pos'], f['train_neg'], f['test_pos'], f['test_neg']
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_pos, train_neg, test_pos, test_neg = sample_dyn(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)


'''Train and apply classifier'''
A = net.copy()  # the observed network
# A[test_pos[0], test_pos[1]] = 0  # mask test links
# A[test_pos[1], test_pos[0]] = 0  # mask test links

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

if not path.exists('data_sta/'+args.split+'h'+str(args.hop)):
    train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(A, args.window, train_pos_id, train_pos, train_neg_id, train_neg, test_pos_id, test_pos, test_neg_id, test_neg, args.hop, args.max_nodes_per_hop, node_information)
    print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))
    with open('data_sta/'+args.split+'h'+str(args.hop), 'wb') as f:
        pickle.dump([train_graphs, test_graphs, max_n_label], f, protocol=4)
else:
    with open('data_sta/'+args.split+'h'+str(args.hop), 'rb') as f:
        train_graphs, test_graphs, max_n_label = pickle.load(f)
        print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))



# DGCNN configurations
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = 20
cmd_args.learning_rate = 1e-4
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
cmd_args.window = 5
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    A = []
    for i in train_graphs:
        #print(type(i[-1]))
        A.append(i[-1])
    for i in test_graphs:
        A.append(i[-1])   
    #print(type(A[0]))
    num_nodes_list = sorted([g.num_nodes for g in A])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print(('k used in SortPooling is: ' + str(cmd_args.sortpooling_k)))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_auc = 0
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2])))

    classifier.eval()
    test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print(('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f avg_precision %.5f precision-recall auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4])))
    if test_loss[2] > best_auc:
        best_auc = test_loss[2]

print('best_auc = ', best_auc)

# with open('acc_results.txt', 'a+') as f:
#     f.write(str(test_loss[1]) + '\n')

# if cmd_args.printAUC:
#     with open('auc_results.txt', 'a+') as f:
#         f.write(str(test_loss[2]) + '\n')

