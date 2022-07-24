# STL
import os
import pickle
import argparse
# 3rd party library
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch

def graph_connect(path, edge_num, adj_save_path, file_save_path, in_cluster_num=25, n_cluster=100):
    all_data = torch.load(path)
    ret = []
    for data in tqdm(all_data):
        cur_ret = {}
        save_path, miss_node, target, in_node_num = data['save_path'], data['miss_node'], data['target'], data['in_node_num']
        cur_ret['save_path'] = save_path
        cur_ret['miss_node'] = miss_node
        cur_ret['target'] = target
        only_name, _ = os.path.splitext(save_path)
        only_name = only_name.split('/')[-1]
        adj_save_name = os.path.join(adj_save_path, only_name + '.npy')
        cur_ret['adj'] = adj_save_name
        cur_ret['in_node_num'] = in_node_num
        ret.append(cur_ret)

        fea = np.load(save_path)
        a, b, c = fea.shape
        assert a == n_cluster
        assert b == in_cluster_num
        fea_for_dis = fea.reshape(a * b, c)

        fea_dis = euclidean_distances(fea_for_dis)
        cur_adj = np.zeros((a * b, a * b))
        for j, dis in enumerate(fea_dis):
            sub_bag = j // in_cluster_num
            if sub_bag in miss_node:
                continue
            if np.all(fea_for_dis[j, :] == 0):
                continue
            dis_index = [[d, i] for i, d in enumerate(dis)]
            dis_index.sort()
            count = 0

            for d, i in dis_index:
                cur_sub_bag = i // in_cluster_num
                if cur_sub_bag in miss_node:
                    continue
                if np.all(fea_for_dis[i, :] == 0):
                    continue
                if cur_sub_bag == sub_bag:
                    cur_adj[j][i] = 1
                else:
                    if count >= edge_num:
                        continue
                    else:
                        cur_adj[j][i] = 1
                        count += 1

        assert not os.path.isfile(adj_save_name)
        np.save(adj_save_name, cur_adj)

    torch.save(ret, file_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Node-aligned graph construction")
    parser.add_argument('--exp_root_path', type=str, help="Experiment root path")
    parser.add_argument('--N_global_cluster', type=int, default=100, help="global cluster number")
    parser.add_argument('--N_local_cluster', type=int, default=15, help="number of locally sampled instances")
    parser.add_argument('--edge_num', type=int, default=5, help="Edge connection numbers")

    args = parser.parse_args()
    print("Experiment root path: {}.".format(args.exp_root_path))

    print("Perform graph construction...")
    train_data_path = os.path.join(args.exp_root_path, "splits", "train.pth")
    file_save_path = os.path.join(args.exp_root_path, "graph", "train_adj.pth")

    adj_save_path = os.path.join(args.exp_root_path, "graph", "fea_adj")
    os.makedirs(adj_save_path, exist_ok=True)

    graph_connect(train_data_path, edge_num=args.edge_num, adj_save_path=adj_save_path, file_save_path=file_save_path,
                     in_cluster_num=args.N_local_cluster, n_cluster=args.N_global_cluster)

    test_data_path = os.path.join(args.exp_root_path, "splits", "test.pth")
    file_save_path = os.path.join(args.exp_root_path, "graph", "test_adj.pth")
    graph_connect(test_data_path, edge_num=args.edge_num, adj_save_path=adj_save_path, file_save_path=file_save_path,
                     in_cluster_num=args.N_local_cluster, n_cluster=args.N_global_cluster)