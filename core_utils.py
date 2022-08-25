# STL
import os
import argparse
# 3rd party library
import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
# local library
from model.gcn import GCN, GCN_Binary


def load_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2_reg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    return optimizer, scheduler


def save_model(args, model):
    out = args.weight_save_path

    dic = {'state': model.state_dict(),
           'lr': args.lr,
           'cluster_num': args.cluster_num,
           'in_cluster_num': args.in_cluster_num}

    torch.save(dic, out)


def confusion_matrix(prediction, target, confusion_m):
    stacked = torch.stack((target, prediction), dim=1)
    for p in stacked:
        t1, p1 = p.tolist()
        confusion_m[t1, p1] += 1

def auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    lbl_true = np.concatenate(lbl_true_list, axis=0)
    lbl_pred = np.concatenate(lbl_pred_list, axis=0)

    if multi_class:
        aucs = []
        binary_labels = label_binarize(lbl_true, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in lbl_true:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], lbl_pred[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        return np.nanmean(np.array(aucs))
    else:
        return roc_auc_score(lbl_true, lbl_pred)



def train(args, train_loader, model, criterion, optimizer):
    model.train()
    loss_epoch = 0
    correct = 0
    total = 0
    for step, (x, y, adj, mask, in_node_num, groups) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.cuda()
        target = y.cuda()
        adj = adj.cuda()
        mask = mask.cuda()
        in_node_num = in_node_num.cuda()
        groups = groups.cuda()

        output = model(x, adj, mask, in_node_num, groups)


        if args.bce == 0:
            prediction = torch.argmax(output, 1)
        else:
            target = target.unsqueeze(1)
            prediction = (output > args.thresh).float()
        correct += (prediction == target).sum().float()
        total += len(target)

        if args.bce == 0:
            loss = criterion(output, target)
        else:
            loss = criterion(output, target.float())
        loss.backward()

        optimizer.step()

        if step % 1 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        loss_epoch += loss.item()
    train_acc = (correct / total).cpu().detach().data.numpy()
    acc_str = 'Train accuracy: %f' % (train_acc)
    print(acc_str)
    return loss_epoch, train_acc


def eval(args, test_loader, model):
    correct = 0
    total = 0
    confusion_m = torch.zeros(args.nclass, args.nclass)

    model.eval()
    lbl_true_list = []
    lbl_pred_list = []
    with torch.no_grad():
        for step, (x, y, adj, mask, in_node_num, groups) in enumerate(test_loader):
            x = x.cuda()

            # print(x.shape)
            target = y.cuda()
            adj = adj.cuda()
            mask = mask.cuda()
            in_node_num = in_node_num.cuda()
            groups = groups.cuda()

            output = model(x, adj, mask, in_node_num, groups)

            if args.nclass == 2:
                if args.bce == 0:
                    prediction = torch.argmax(output, 1)
                    lbl_true_list.append(target.detach().cpu().numpy())
                    lbl_pred_list.append(output[:, 1].detach().cpu().numpy())
                else:
                    target = target.unsqueeze(1)
                    prediction = (output > args.thresh).long()
                    lbl_true_list.append(target.squeeze().detach().cpu().numpy())
                    lbl_pred_list.append(output[:, 0].detach().cpu().numpy())
            else:
                assert args.nclass > 2
                prediction = torch.argmax(output, 1)
                lbl_true_list.append(target.detach().cpu().numpy())
                lbl_pred_list.append(output.detach().cpu().numpy())

            confusion_matrix(prediction, target, confusion_m)

            correct += (prediction == target).sum().float()
            total += len(target)

    acc = (correct / total).cpu().detach().data.numpy()
    auc_score = auc(lbl_true_list, lbl_pred_list, multi_class=args.nclass > 2, n_classes=args.nclass)

    print(confusion_m)
    acc_str = 'Test accuracy: {}, Test AUC: {}'.format(acc, auc_score)
    print(acc_str)
    return acc, auc_score


def main(args, train_loader, test_loader):
    torch.manual_seed(0)
    np.random.seed(0)

    if args.bce == 0:
        print("Use GCN.")
        model = GCN(nfeat=args.nfeat, nclass=args.nclass, node_nums=args.node_nums, cluster_num=args.cluster_num, dropout=args.dropout)
    else:
        if not args.vanilla:
            print("Use GCN_Binary.")
            model = GCN_Binary(nfeat=args.nfeat, node_nums=args.node_nums, cluster_num=args.cluster_num,
                          dropout=args.dropout)
        else:
            pass


    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    if args.bce == 0:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss()
    criterion = criterion.cuda()


    if torch.cuda.is_available():
        model = model.cuda()


    args.global_step = 0
    args.current_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, train_acc = train(args, train_loader, model, criterion, optimizer)
        if scheduler:
            scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {lr}"
        )
        args.current_epoch += 1

    save_model(args, model)
    test_acc, auc_score = eval(args, test_loader, model)

    return loss_epoch / len(train_loader), test_acc, auc_score


if __name__ == "__main__":
    pass
