# STL
import os
import pickle
# 3rd party library
import numpy as np
import torch
import torch.utils.data as Data


def read_data(data_paths, cluster_num):
    imgs_path = []
    adjs_path = []
    target_list = []
    miss_node_list = []
    in_node_list = []
    for data in data_paths:
        data_path, miss_node, target, adj, in_node_num = data['save_path'], data['miss_node'], data['target'], data['adj'], data['in_node_num']

        imgs_path.append(data_path)
        target_list.append(target)
        adjs_path.append(adj)

        in_node = np.zeros(cluster_num)
        for node, num in in_node_num.items():
            if num != 0:
                in_node[node] = 1 / num
        in_node_list.append(in_node)

        mask = np.ones(cluster_num)
        for num in miss_node:
            mask[num] = 0
        miss_node_list.append(mask)

    return imgs_path, target_list, adjs_path, miss_node_list, in_node_list


class NAGCN_Dataset(Data.Dataset):
    def __init__(self, data_paths, target2index, node_nums, in_nums, cluster_num, fea_dim):
        self.imgs_path, self.target_list, self.adjs_path, self.miss_node_list, self.in_node_list = read_data(
            data_paths, cluster_num)
        self.target2index = target2index
        self.groups = torch.tensor([i // in_nums for i in range(node_nums)]).float()
        self.node_nums = node_nums
        self.in_nums = in_nums
        self.fea_dim = fea_dim
        if os.path.splitext(self.imgs_path[0])[-1] == '.npy':
            self.npy = True
        else:
            assert os.path.splitext(self.imgs_path[0])[-1] == '.pickle'
            self.npy = False

    def __getitem__(self, index):
        if self.npy:
            cur_data = np.load(self.imgs_path[index])
        else:
            with open(self.imgs_path[index], 'rb') as file:
                cur_data = pickle.load(file)

        # sampled WSI features
        cur_data = cur_data.reshape(self.node_nums, self.fea_dim)
        cur_data = torch.from_numpy(cur_data)
        cur_data = cur_data.float()

        if self.npy:
            cur_adj = np.load(self.adjs_path[index])
        else:
            with open(self.adjs_path[index], 'rb') as file:
                cur_adj = pickle.load(file)

        # adjacent matrix
        cur_adj = torch.from_numpy(cur_adj)
        cur_adj = cur_adj.float()
        cur_target = self.target_list[index]
        target = torch.tensor(self.target2index[cur_target])

        mask = torch.from_numpy(self.miss_node_list[index])
        mask = mask.float()

        in_node_num = torch.from_numpy(self.in_node_list[index])
        in_node_num = in_node_num.float()

        groups = self.groups

        return cur_data, target, cur_adj, mask, in_node_num, groups

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    pass
