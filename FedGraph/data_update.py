#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import json
import torch
import concepts
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from torch_sparse import coalesce
from torch_geometric.data import Data
from sklearn import model_selection
# import stellargraph as sg
from sklearn import model_selection


def PowerSetsBinary(items):
    N = len(items)
    z = []
    for i in range(2 ** N):
        zj = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                zj.append(items[j])
        # print(zj)
        z.append(zj)
    return z


def file_convert(args):
    a = open("data/concept/{}_edge.csv".format(args.dataset), 'r')
    reader = csv.reader(a)
    with open("data/concept/{}_edge.txt".format(args.dataset), 'w') as f:
        for i in reader:
            for x in i:
                f.write(x)
                f.write('\t')
            f.write('\n')
    a.close()

    # b = open("../concept/{}_context.csv".format(args.dataset), 'r')
    # reader = csv.reader(b)
    # with open("../concept/{}_text.txt".format(args.dataset), 'w') as f:
    #     for i in reader:
    #         for x in i:
    #             f.write(x)
    #             f.write('\t')
    #         f.write('\n')
    # b.close()


def read_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    file.close()
    return data


def PowerSetsInsert(data):
    # if len(data) == 2:
    #     return [val for val in data[0] if val in data[1]]
    # else:
    #     return [val for val in PowerSetsInsert(data[0:len(data)-1]) if val in data[-1]]
    res = data[0]
    for i in range(1, len(data)):
        res = [val for val in res if val in data[i]]
    return res


def load_lastfm():
    lastfm_asia_edges = pd.read_csv("lastfm_asia/lastfm_asia_edges.csv", sep=',')
    G = nx.from_pandas_edgelist(lastfm_asia_edges, 'node_1', 'node_2')
    with open("lastfm_asia/lastfm_asia_features.json", 'r') as f:
        lastfm_asia_features = json.load(f)

    nodes = []
    for node in G.nodes():
        nodes.append(node)
    nodes.sort()
    df_node = pd.DataFrame(nodes)

    features = []
    max_value = 0
    for feature in lastfm_asia_features.values():
        features.append(feature)
        for feat in feature:
            if feat > max_value:
                max_value = feat
    feature01 = np.zeros((len(features), max_value+1), dtype=int)
    for i in range(len(features)):
        for value in features[i]:
            feature01[i, value] = 1

    df_feature = pd.DataFrame(feature01)

    lastfm_asia_label = pd.read_csv("lastfm_asia/lastfm_asia_target.csv", sep=',')
    df_label = lastfm_asia_label[lastfm_asia_label.columns[-1]]
    node_subjects = pd.concat([df_node, df_feature, df_label], axis=1)

    return G, node_subjects


def load_facebook_large():
    facebook_edges = pd.read_csv("facebook_large/facebook_edges.csv", sep=',')
    G = nx.from_pandas_edgelist(facebook_edges, 'id_1', 'id_2')
    with open("facebook_large/facebook_features.json", 'r') as f:
        facebook_features = json.load(f)
    nodes = []
    for node in G.nodes():
        nodes.append(node)
    nodes.sort()
    df_node = pd.DataFrame(nodes)

    features = []
    max_value = 0
    for feature in facebook_features.values():
        features.append(feature)
        for feat in feature:
            if feat > max_value:
                max_value = feat
    feature01 = np.zeros((len(features), max_value+1), dtype=int)
    for i in range(len(features)):
        for value in features[i]:
            feature01[i, value] = 1
    df_feature = pd.DataFrame(feature01)

    facebook_label = pd.read_csv("facebook_large/facebook_target.csv", sep=',')
    df_label = facebook_label[facebook_label.columns[-1]]
    node_subjects = pd.concat([df_node, df_feature, df_label], axis=1)

    return G, node_subjects


def load_reddit():
    reddit = np.load("reddit_2/reddit_data.npz", 'r', allow_pickle=True)

    reddit_graph = np.load("reddit_2/reddit_graph.npz", 'r', allow_pickle=True)
    row = reddit_graph['row']
    col = reddit_graph['col']
    G = nx.Graph()
    for i in range(len(row)):
        for j in range(len(col)):
            G.add_edge(row[i], col[j])
    nodes = []
    for node in G.nodes():
        nodes.append(node)
    nodes.sort()
    df_node = pd.DataFrame(nodes)

    reddit_features = reddit['feature']
    df_feature = pd.DataFrame(reddit_features)

    reddit_labels = reddit['label']
    df_label = pd.DataFrame(reddit_labels)
    node_subjects = pd.concat([df_feature, df_label], axis=1)

    return G, node_subjects


def load_1():
    node_subjects = pd.read_csv("2/2.content", sep='\t', header=None)
    edges = pd.read_csv("2/2.cites", sep=',', header=None)
    columns = ['id1', 'id2']
    edges = edges.rename(columns={i: c for i, c in enumerate(columns)})
    G = nx.from_pandas_edgelist(edges, 'id1', 'id2')

    return G, node_subjects



def update_data(dataset):

    print('Begin')
    # 获取数据集
    # datasets = getattr(sg.datasets, dataset)  # dataset = sg.datasets.Cora
    G, node_subjects = load_facebook_large()

    len_data = G.number_of_nodes()
    df_feature_origin = node_subjects.iloc[:, 1:-1]
    print('数据集获取成功')
    # 生成形式背景

    A = np.zeros((len_data + 1, len_data + 1)).tolist()

    names = [f"a{i}" for i in range(len_data)]
    for i in range(1, len_data + 1):
        A[i][0] = i
        A[0][i] = names[i-1]
    for i in range(0, len_data):
        for j in range(0, len_data):
            if G.has_edge(i, j):
                A[i + 1][j + 1] = 1
    # for i in range(1, len_data + 1):
    #     A[i][i] = 1
    for i in range(1, len_data + 1):
        for j in range(1, len_data + 1):
            if A[i][j] == 0:
                A[i][j] = " "
            else:
                A[i][j] = "X"
    with open("facebook_large/{}_context.csv".format(dataset), mode="w", encoding="utf-8", newline="") as f1:
        writer1 = csv.writer(f1)
        writer1.writerows(A)
    print('形式背景已保存')

    # 生成概念格
    c = concepts.load_csv("facebook_large/{}_context.csv".format(dataset), encoding='utf-8')
    print('概念格生成完成')
    # 保存概念信息
    la = c.lattice
    data_concept = []
    data_equiconcept = []
    for extent, intent in la:
        print(extent, intent)
        data_concept.append([extent, intent])
        str_ex = ''
        str_in = ''
        for i in extent:
            str_ex += i
        for j in intent:
            str_in += j[1]
        if str_ex == str_in:
            data_equiconcept.append([extent, intent])
    df_equiconcept = pd.DataFrame(data_equiconcept)

    # 特征更新
    feature_new = np.zeros((len_data, df_equiconcept.shape[0]), dtype=int)
    for i in range(df_equiconcept.shape[0]):
        extent_equip = df_equiconcept.iloc[i][0]
        for j in extent_equip:
            feature_new[int(j) - 1][i] = 1

    feature_origin = np.zeros((df_feature_origin.shape[0], df_feature_origin.shape[1]), dtype=int)
    for i in range(df_feature_origin.shape[0]):
        for j in range(df_feature_origin.shape[1]):
            feature_origin[i][j] = df_feature_origin.iloc[i, j]
    df_feature_new = pd.DataFrame(feature_new)
    df_feature_new.to_csv("facebook_large/feature_new.csv", header=False, index=False)
    print('END!!!')


def update(data):
    feat = pd.read_csv('facebook_large/feature_new.csv', header=None)
    feature = np.zeros((feat.shape[0], feat.shape[1]), dtype=int)
    for i in range(feat.shape[0]):
        for j in range(feat.shape[1]):
            feature[i][j] = feat.iloc[i, j]
    feature = torch.Tensor(feature)

    torch.save(feature, 'facebook_large/feature.pt')
    x_new = torch.load('facebook_large/feature.pt')
    data.x = torch.cat([data.x, x_new], dim=1)
    return data.x


def lastfm_to_Tensor():
    lastfm_asia_edges = pd.read_csv("lastfm_asia/lastfm_asia_edges.csv", sep=',')
    edges = lastfm_asia_edges.to_numpy()
    with open("lastfm_asia/lastfm_asia_features.json", 'r') as f:
        lastfm_asia_features = json.load(f)
    features = []
    max_value = 0
    for feature in lastfm_asia_features.values():
        features.append(feature)
        for feat in feature:
            if feat > max_value:
                max_value = feat
    feature01 = np.zeros((len(features), max_value+1), dtype=int)
    for i in range(len(features)):
        for value in features[i]:
            feature01[i, value] = 1

    lastfm_asia_label = pd.read_csv("lastfm_asia/lastfm_asia_target.csv", sep=',')
    df_label = lastfm_asia_label[lastfm_asia_label.columns[-1]]
    label = df_label.to_numpy()

    x = torch.from_numpy(feature01).to(torch.float)
    y = torch.from_numpy(label).to(torch.long)
    edge_index = torch.from_numpy(edges).to(torch.long)
    edge_index = edge_index.t().contiguous()
    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def facebook_large_to_Tensor():
    facebook_edges = pd.read_csv("facebook_large/musae_facebook_edges.csv", sep=',')
    edges = facebook_edges.to_numpy()
    G = nx.from_pandas_edgelist(facebook_edges, 'id_1', 'id_2')
    with open("facebook_large/musae_facebook_features.json", 'r') as f:
        facebook_features = json.load(f)
    nodes = []
    for node in G.nodes():
        nodes.append(node)
    nodes.sort()
    df_node = pd.DataFrame(nodes)

    features = []
    max_value = 0
    for feature in facebook_features.values():
        features.append(feature)
        for feat in feature:
            if feat > max_value:
                max_value = feat
    feature01 = np.zeros((len(features), max_value+1), dtype=int)
    for i in range(len(features)):
        for value in features[i]:
            feature01[i, value] = 1
    df_feature = pd.DataFrame(feature01)

    facebook_label = pd.read_csv("facebook_large/musae_facebook_target.csv", sep=',')
    df_label = facebook_label[facebook_label.columns[-1]]
    label = df_label.to_numpy()

    x = torch.from_numpy(feature01).to(torch.float)
    y = torch.from_numpy(edges).to(torch.long)

    edge_index = torch.from_numpy(np.array(G.edges())).to(torch.long)
    edge_index = edge_index.t().contiguous()
    data = Data(x=x, y=y, edge_index=edge_index)

    return data





def  facebook_to_Tensor():
    facebook_edges = pd.read_csv("facebook_large/facebook_edges.csv", sep=',')
    edges = facebook_edges.to_numpy()
    G = nx.from_pandas_edgelist(facebook_edges, 'id_1', 'id_2')
    with open("facebook_large/facebook_features.json", 'r') as f:
        facebook_features = json.load(f)
    nodes = []
    for node in G.nodes():
        nodes.append(node)
    nodes.sort()
    df_node = pd.DataFrame(nodes)

    features = []
    max_value = 0
    for feature in facebook_features.values():
        features.append(feature)
        for feat in feature:
            if feat > max_value:
                max_value = feat
    feature01 = np.zeros((len(features), max_value+1), dtype=int)
    for i in range(len(features)):
        for value in features[i]:
            feature01[i, value] = 1
    df_feature = pd.DataFrame(feature01)

    facebook_label = pd.read_csv("facebook_large/facebook_target.csv", sep=',')
    df_label = facebook_label[facebook_label.columns[-1]]
    label = df_label.to_numpy()

    x = torch.from_numpy(feature01).to(torch.float)
    y = torch.from_numpy(edges).to(torch.long)

    edge_index = torch.from_numpy(np.array(G.edges())).to(torch.long)
    edge_index = edge_index.t().contiguous()
    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def reddit_to_Tensor():
    data = np.load('reddit_2/reddit_data.npz')
    x = torch.from_numpy(data['feature']).to(torch.float)
    y = torch.from_numpy(data['label']).to(torch.long)
    print(x)
    adj = sp.load_npz('reddit_2/reddit_graph.npz')
    print(adj.row)
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def To_Tensor():
    G, node_subjects = load_1()
    feature = node_subjects.iloc[:, 1:-1]
    label = node_subjects.iloc[:, -1]
    edges = pd.read_csv("1/1.cites", sep=',')
    edge = edges.to_numpy()
    x = torch.from_numpy(feature.values).to(torch.float)
    y = torch.from_numpy(label.values).to(torch.long)
    edge_index = torch.from_numpy(edge).to(torch.long)
    edge_index = edge_index.t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=y)

    return data
