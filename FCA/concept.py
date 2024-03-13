#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import itertools
import concepts
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


def load_data1():
    edges = pd.read_csv('citeseer-doc-classification/citeseer.cites', header=None, sep='\t')
    edges.columns = ['node_1', 'node_2']
    G = nx.from_pandas_edgelist(edges, 'node_1', 'node_2')
    node_subjects = pd.read_csv('citeseer-doc-classification/citeseer.content', header=None, sep='\t')
    return G, node_subjects




def load_data2():
    edgelist = pd.read_csv(
        'pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab',
        sep="\t",
        skiprows=2,
        header=None,
        names=["id", "source", "pipe", "target"],
        usecols=["source", "target"],
    )
    edgelist.source = edgelist.source.str.lstrip("paper:").astype(int)
    edgelist.target = edgelist.target.str.lstrip("paper:").astype(int)
    G = nx.from_pandas_edgelist(edgelist, 'source', 'target')

    with open('pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab') as fp:
        node_data = pd.DataFrame(
            parse_line(line) for line in itertools.islice(fp, 2, None)
        )

    node_data.fillna(0, inplace=True)
    node_data.set_index("pid", inplace=True)

    # labels = node_data["label"]
    #
    # nodes = node_data.drop(columns="label")
    return G, node_data


def parse_feature(feat):
    name, value = feat.split("=")
    return name, float(value)


def parse_line(line):
    pid, raw_label, *raw_features, _summary = line.split("\t")
    features = dict(parse_feature(feat) for feat in raw_features)
    features["pid"] = int(pid)
    features["label"] = int(parse_feature(raw_label)[1])
    return features


def update_data():

    print('Begin')
    # 获取数据集
    # datasets = getattr(sg.datasets, dataset)  # dataset = sg.datasets.Cora
    G, node_subjects = load_data1()
    # train_subjects, test_subjects = model_selection.train_test_split(
    #     node_subjects, train_size=10, test_size=None, stratify=node_subjects
    # )

    print('数据集获取成功')
    # 生成形式背景
    matrix = nx.to_numpy_matrix(G)
    rows, cols = matrix.shape
    A = np.zeros((rows + 1, cols + 1)).tolist()
    names = [f"a{i}" for i in range(rows)]
    for i in range(1, rows + 1):
        A[i][0] = i
        A[0][i] = names[i - 1]
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                A[i + 1][j + 1] = "X"
            else:
                A[i + 1][j + 1] = " "
    for i in range(1, rows + 1):
        A[i][i] = "X"
    with open("citeseer-doc-classification/context.csv", mode="w", encoding="utf-8", newline="") as f1:
        writer1 = csv.writer(f1)
        writer1.writerows(A)
    print('形式背景已保存')

    # 生成概念格
    c = concepts.load_csv("citeseer-doc-classification/context.csv", encoding='utf-8')
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
    feature_new = np.zeros((rows, df_equiconcept.shape[0]), dtype=int)
    for i in range(df_equiconcept.shape[0]):
        extent_equip = df_equiconcept.iloc[i][0]
        for j in extent_equip:
            feature_new[int(j) - 1][i] = 1
    df_feature_new = pd.DataFrame(feature_new)
    df_feature_new.to_csv("citeseer-doc-classification/feature_new.csv", header=False, index=False)
    print('END!!!')

def load_1():
    node_subjects = pd.read_csv("1.content", sep='\t', header=None)
    edges = pd.read_csv("1.cites", sep=',', header=None)
    columns = ['id1', 'id2']
    edges = edges.rename(columns={i: c for i, c in enumerate(columns)})
    G = nx.from_pandas_edgelist(edges, 'id1', 'id2')

    return G, node_subjects
def To_Tensor():
    G, node_subjects = load_1()
    feature = node_subjects.iloc[:, 1:-1]
    label = node_subjects.iloc[:, -1]
    edges = pd.read_csv("1.cites", sep=',')
    edge = edges.to_numpy()
    x = torch.from_numpy(feature.values).to(torch.float)
    y = torch.from_numpy(label.values).to(torch.long)
    edge_index = torch.from_numpy(edge).to(torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    return data




