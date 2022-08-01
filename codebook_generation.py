# STL
import os
import random
import argparse
# 3rd party library
from tqdm import tqdm
import numpy as np
import pandas as pds
from sklearn.cluster import MiniBatchKMeans, KMeans
import torch



def get_cluster(datas, k, num=-1, seed=None):
    if seed is not None:
        random.seed(seed)
    if num > 0:
        data_sam = random.sample(datas, num)
    else:
        data_sam = datas
    if seed is not None:
        random.seed(seed)
    random.shuffle(data_sam)

    dim_N = 0
    dim_D = data_sam[0]['shape'][1]
    for data in tqdm(data_sam):
        dim_N += data['shape'][0]
    con_data = np.zeros((dim_N, dim_D), dtype=np.float32)
    ind = 0
    for data in tqdm(data_sam):
        data_path, data_shape = data['slide'], data['shape']
        cur_data = torch.load(data_path)
        con_data[ind:ind + data_shape[0], :] = cur_data.numpy()
        ind += data_shape[0]
    # clusterer = KMeans(n_clusters=k)
    clusterer = MiniBatchKMeans(n_clusters=k, batch_size=10000)
    clusterer.fit(con_data)
    print("cluster done")
    return clusterer

def get_data(datas):
    index2name = []
    all_name = []
    target = {}
    dim_N = 0
    dim_D = datas[0]['shape'][1]
    for data in tqdm(datas):
        dim_N += data['shape'][0]
    con_data = np.zeros((dim_N, dim_D), dtype=np.float32)

    ind = 0
    for i, data in tqdm(enumerate(datas)):
        data_path, data_shape, data_target = data['slide'], data['shape'], data['target']
        only_name, _ = os.path.splitext(data_path)
        only_name = only_name.split('/')[-1]
        cur_data = torch.load(data_path)
        con_data[ind:ind + data_shape[0], :] = cur_data.numpy()
        ind += data_shape[0]
        index2name.extend([only_name] * int(data_shape[0]))
        target[only_name] = data_target
        all_name.append(only_name)

    return con_data, index2name, all_name, target

def train_test_split(data_list, df):
    train_data = []
    test_data = []
    for data in data_list:
        name, _ = os.path.splitext(data["slide"])
        name = name.split('/')[-1]
        assert name in df.index
        if df.loc[name]["train"]:
            train_data.append(data)
        else:
            test_data.append(data)

    return train_data, test_data



def K_cluster_in_cluster(data, k, index2name, all_name, save_path, targets, clusterer, in_cluster_num=15):

    cluster_label = clusterer.predict(data)
    cluster_fea = {name: np.zeros((k, in_cluster_num, data.shape[1])) for name in all_name}
    num_cluster_fea = {name: [[] for _ in range(k)] for name in all_name}
    miss_fea = {name: set([i for i in range(k)]) for name in all_name}
    in_node_num = {name: {i: 0 for i in range(k)} for name in all_name}

    for i, label in tqdm(enumerate(cluster_label)):
        cur_name = index2name[i]
        num_cluster_fea[cur_name][label].append(i)
        if label in miss_fea[cur_name]:
            miss_fea[cur_name].remove(label)
    for cur_name in tqdm(all_name):
        try:
            for i in range(k):
                cur_node_ind = num_cluster_fea[cur_name][i]
                cur_node_num = len(cur_node_ind)
                if cur_node_num <= in_cluster_num:
                    in_node_num[cur_name][i] = cur_node_num
                    cluster_fea[cur_name][i, :cur_node_num, :] = data[cur_node_ind]
                else:
                    in_node_num[cur_name][i] = in_cluster_num
                    data_sub = data[cur_node_ind].copy()
                    # clser = MiniBatchKMeans(n_clusters=in_cluster_num).fit(data_sub)
                    clser = KMeans(n_clusters=in_cluster_num).fit(data_sub)
                    clser_lbl = clser.labels_
                    for j in range(in_cluster_num):
                        random_ind = np.random.choice(np.where(clser_lbl == j)[0])
                        cluster_fea[cur_name][i, j, :] = data_sub[random_ind]

        except:
            try:
                cluster_fea[cur_name][:, :, :] = 0
                for i in range(k):
                    cur_node_ind = num_cluster_fea[cur_name][i]
                    cur_node_num = len(cur_node_ind)
                    if cur_node_num <= in_cluster_num:
                        in_node_num[cur_name][i] = cur_node_num
                        cluster_fea[cur_name][i, :cur_node_num, :] = data[cur_node_ind]
                    else:
                        in_node_num[cur_name][i] = in_cluster_num
                        data_sub = data[cur_node_ind].copy()
                        # clser = MiniBatchKMeans(n_clusters=in_cluster_num).fit(data_sub)
                        clser = KMeans(n_clusters=in_cluster_num).fit(data_sub)
                        clser_lbl = clser.labels_
                        for j in range(in_cluster_num):
                            random_ind = np.random.choice(np.where(clser_lbl == j)[0])
                            cluster_fea[cur_name][i, j, :] = data_sub[random_ind]
            except:
                cluster_fea[cur_name][:, :, :] = 0
                for i in range(k):
                    cur_node_ind = num_cluster_fea[cur_name][i]
                    cur_node_num = len(cur_node_ind)
                    if cur_node_num <= in_cluster_num:
                        in_node_num[cur_name][i] = cur_node_num
                        cluster_fea[cur_name][i, :cur_node_num, :] = data[cur_node_ind]
                    else:
                        in_node_num[cur_name][i] = in_cluster_num
                        data_sub = data[cur_node_ind].copy()
                        # clser = MiniBatchKMeans(n_clusters=in_cluster_num).fit(data_sub)
                        clser = KMeans(n_clusters=in_cluster_num).fit(data_sub)
                        clser_lbl = clser.labels_
                        for j in range(in_cluster_num):
                            random_ind = np.random.choice(np.where(clser_lbl == j)[0])
                            cluster_fea[cur_name][i, j, :] = data_sub[random_ind]

    torch_file = []
    for name, data in tqdm(cluster_fea.items()):
        save_fold = os.path.join(save_path, name)
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)
        save_name = os.path.join(save_fold, name + '.npy')
        np.save(save_name, data)
        torch_file.append(
            {
                "save_path": save_name,
                "miss_node": miss_fea[name],
                "in_node_num": in_node_num[name],
                "target": targets[name]}
        )

    return torch_file




def cluster_sample(data_paths, k, save_path, train_file_path, test_file_path, df, instance_num, test_in_num=15, train_in_num = 15):
    '''
    perform cluster sampling
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    all_data = []
    for path in data_paths:
        d = torch.load(path)
        all_data.append(d)
    cluster_data = []
    for i, data in enumerate(all_data):
        cluster_data.extend(data)
    train_data, test_data = train_test_split(cluster_data, df)

    clusterer = get_cluster(train_data, k, num=instance_num, seed=3)
    con_data, index2name, all_name, target = get_data(train_data)
    cur_res = K_cluster_in_cluster(con_data, k, index2name, all_name, save_path, target, clusterer, in_cluster_num=train_in_num)
    torch.save(cur_res, train_file_path, _use_new_zipfile_serialization=False)
    con_data, index2name, all_name, target = get_data(test_data)
    cur_res = K_cluster_in_cluster(con_data, k, index2name, all_name, save_path, target, clusterer, in_cluster_num=test_in_num)
    torch.save(cur_res, test_file_path, _use_new_zipfile_serialization=False)




def get_parser():
    parser = argparse.ArgumentParser(description="Hierarchical global-to-local clustering sampling.")

    parser.add_argument('--fea_path', type=str)  # index list
    parser.add_argument('--instance_num', type=int, default=-1)
    parser.add_argument('--N_global_cluster', type=int, default=100)
    parser.add_argument('--train_in_num', type=int, default=15)
    parser.add_argument('--test_in_num', type=int, default=15)
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--exp_root_path', type=str)
    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    data_paths = []
    data_paths.append(args.fea_path)

    os.makedirs(args.exp_root_path, exist_ok=True)
    os.makedirs(os.path.join(args.exp_root_path, "splits"), exist_ok=True)


    df_splits = pds.read_csv(args.splits_path, index_col=0)
    fea_save_path = os.path.join(args.exp_root_path, "cluster_fea")
    os.makedirs(fea_save_path, exist_ok=True)
    train_file_path = os.path.join(args.exp_root_path, "splits", "train.pth")
    test_file_path = os.path.join(args.exp_root_path, "splits", "test.pth")

    cluster_sample(data_paths, args.N_global_cluster, fea_save_path, train_file_path, test_file_path, df=df_splits, instance_num=args.instance_num, test_in_num=args.test_in_num, train_in_num=args.train_in_num)