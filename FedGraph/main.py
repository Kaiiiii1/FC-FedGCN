import sys
import time
import numpy as np
from sklearn import model_selection

sys.path.append('/home/lames/code/Gcode')
from utils import *
from data_loader import get_data, load_partition_data
from Arguments import Arguments
from client import clientSP
from trainersp import TrainerSP
import argparse
from gcnlp import GCNLP
from sagelp import SageLP
from gatlp import GATLP
import torch_geometric.transforms as T
import logging
import torch
from sklearn.metrics import roc_auc_score


def init_args():
    args = {}
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('-n', '--clientnum', type=int)
    parser.add_argument('-r', '--round', type=int)
    parser.add_argument('-e', '--epoch', type=int)
    parser.add_argument('-d', '--name', type=str)
    x = parser.parse_args()
    args["client_num"] = x.clientnum
    args["comm_round"] = x.round
    args["epoch"] = x.epoch
    args["lr"] = 0.01
    args["worker_num"] = x.clientnum
    args["name"] = x.name
    Args = Arguments(args)

    return Args


def agg(client_num_list, model_params_list):
    """
    模型聚合
    """
    sum = 0
    for i in client_num_list:
        sum += i

    params = model_params_list[0]

    for key in params.keys():
        for i in range(len(model_params_list)):
            model_params = model_params_list[i]
            if i == 0:
                params[key] = model_params[key] * client_num_list[i] / sum
            else:
                params[key] += model_params[key] * client_num_list[i] / sum

    return params


def compute_ap(pred, true):
    idx = torch.argsort(pred, descending=True)
    true = true[idx]
    tp = torch.cumsum(true, dim=0)
    fp = torch.cumsum(1 - true, dim=0)
    recall = tp / (true.sum() + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap.item()


if __name__ == '__main__':
    s_time = time.time()
    args = init_args()
    device = 'cuda'
    round = args.comm_round
    epoch = args.epoch
    clientnum = args.client_num
    name = args.name
    data = get_data(name)

    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )

    train_data, val_data, test_data = split(data)
    logging.info('{} ss'.format(round))
    all_model = None
    client_num_list = []
    clientdata_list = []
    acc_list = []
    macro_list = []
    auc_list = []
    _, node_lists = load_partition_data(clientnum, data)
    data.to(device)

    sum = 0
    for nodes in node_lists:
        client_num_list.append(len(nodes))
        sum += len(nodes)

    # 定义模型
    server_model = GCNLP(data.num_node_features, 128, 64).to(device)
    model = GCNLP(data.num_node_features, 128, 64).to(device)
    # 定义trainer
    trainer = TrainerSP(args=args, model=model, data=None)
    # 定义client
    client = clientSP(args=args, trainer=trainer)

    for i in range(round):
        logging.info("开始第 {} 轮训练".format(i))
        local_model = server_model.state_dict()  # 客户端本地模型
        all_model = None  # 清空
        for i in range(clientnum):
            clientdata = get_data_lp_withoutlink(data, train_data, i + 1, node_lists)
            client.trainer.set_data(clientdata)
            client.trainer.set_model(local_model)
            # 本地训练
            model_params, _ = client.train()
            # 聚合操作
            if all_model == None:
                all_model = model_params
            for key in model_params.keys():
                if i == 0:
                    all_model[key] = model_params[key] * client_num_list[i] / sum
                else:
                    all_model[key] += model_params[key] * client_num_list[i] / sum
        # 更新全局模型
        server_model.load_state_dict(all_model)
        server_model.eval()
        test_data.to(device)
        z = model.encode(test_data)
        out = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
        print(out)
        auc = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())
        auc_list.append(auc)
    max_auc = max(auc_list)
    logging.error(
        'client {}  round {}  epoch {}  name {} acc {} max_acc {}'.format(clientnum, round, epoch, name, auc_list,
                                                                          max_auc))
    e_time = time.time()
    run_time = e_time - s_time
    out = open('output.txt', 'a')
    out.write(str(clientnum))
    out.write(" ")
    out.write(str(epoch))
    out.write(" ")
    out.write(str(round))
    out.write(" ")
    out.write(str(max_auc))
    out.write(" ")
    out.write(str(name))
    out.write(" ")
    out.write(str(run_time))
    out.write("\n")
    out.close()
