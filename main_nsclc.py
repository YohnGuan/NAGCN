# STL
import os
import argparse
# 3rd party library
import torch
# local library
from core_utils import main
from utils.data_set import NAGCN_Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--exp_root_path', type=str)
    parser.add_argument('--weight_save_path', type=str,
                        default="weight.pth")
    parser.add_argument('--log', type=str, default='log.txt')

    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--node_nums', type=int, default=2500)  # total numbers of graph nodes
    parser.add_argument('--cluster_num', type=int, default=100)
    parser.add_argument('--in_cluster_num', type=int, default=25)
    parser.add_argument('--nfeat', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_step', type=list, default=[30, 70])
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bce', type=int, default=1)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument("--vanilla", type=bool, default=False)
    parser.add_argument('--L2_reg', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.4)

    args = parser.parse_args()
    log_description = "Experiment root path: {}\n".format(args.exp_root_path)
    print(log_description)

    target2index = {"lusc": 0, "luad": 1}

    os.makedirs(os.path.join(args.exp_root_path, "save"), exist_ok=True)
    os.makedirs(os.path.join(args.exp_root_path, "weight"), exist_ok=True)
    args.weight_save_path = os.path.join(args.exp_root_path, "weight", args.weight_save_path)

    log = open(os.path.join(args.exp_root_path, "save", args.log), "w")
    log.write(log_description)
    log.flush()
    print('--------TCGA NSCLC args----------')
    log.write('--------TCGA NSCLC args----------\n')
    log.flush()
    for k, v in vars(args).items():
        log_str = "{}: {}".format(k, v)
        log.write(log_str + '\n')
        log.flush()
        print(log_str)
    log.write('--------TCGA NSCLC args----------\n')
    log.flush()
    print('--------TCGA NSCLC args----------\n')




    train_file_path = os.path.join(args.exp_root_path, "graph", "train_adj.pth")
    test_file_path = os.path.join(args.exp_root_path, "graph", "test_adj.pth")
    train_data = torch.load(train_file_path)
    test_data = torch.load(test_file_path)

    train_dataset = NAGCN_Dataset(train_data, target2index, node_nums=args.node_nums,
                                       in_nums=args.in_cluster_num,
                                       cluster_num=args.cluster_num, fea_dim=args.nfeat)
    test_dataset = NAGCN_Dataset(test_data, target2index, node_nums=args.node_nums,
                                      in_nums=args.in_cluster_num,
                                      cluster_num=args.cluster_num, fea_dim=args.nfeat)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    train_loss, test_acc, auc_score = main(args, train_loader, test_loader)

    log_str = "train loss: {}\ntest_acc: {}, auc_score: {}\n".format(train_loss, test_acc, auc_score)
    log.write(log_str)
    log.flush()
    print(log_str)

    log.close()
