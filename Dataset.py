import dgl
import pandas as pd
import torch
import numpy as np

from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if torch.cuda.is_available():
    device = torch.device('cuda:0')

# KIBA   Davis
class Dataset_DTA(Dataset):
    def __init__(self, dataset='Davis', drug_graph=None, drug_embedding=None, compound_id=None, target_graph=None, target_embedding=None, target_id=None, affinity=None):

        self.dataset = dataset
        self.drug_graph, _ = load_graphs(drug_graph)
        self.drug_graph = list(self.drug_graph)
        self.drug_embedding = np.load(drug_embedding, allow_pickle=True)

        self.target_graph, _ = load_graphs(target_graph)
        self.target_graph = list(self.target_graph)
        self.target_embedding = np.load(target_embedding, allow_pickle=True)

        self.compound_id = np.load(compound_id, allow_pickle=True)
        self.target_id = np.load(target_id, allow_pickle=True)
        self.affinity = np.load(affinity, allow_pickle=True)


    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):

        drug_len = self.drug_graph[idx].num_nodes()
        target_len = self.target_embedding[idx].shape[0]     
        return self.drug_graph[idx], self.drug_embedding[idx], self.target_graph[idx], self.target_embedding[idx], drug_len, target_len, self.affinity[idx]

    def collate(self, sample):
        batch_size = len(sample)
	  
        drug_graph, drug_embedding, target_graph, target_embedding, drug_len, target_len, affinity = map(list, zip(*sample))
        max_drug_len = max(drug_len)
        max_target_len = max(target_len)

        for i in range(batch_size):
            if target_embedding[i].shape[0] < max_target_len:
                target_embedding[i] = np.pad(target_embedding[i], ((0, max_target_len-target_embedding[i].shape[0]), (0, 0)), mode='constant', constant_values = (0,0))

        drug_graph = dgl.batch(drug_graph).to(device)
        drug_embedding = torch.FloatTensor(drug_embedding).to(device)

        target_graph = dgl.batch(target_graph).to(device)
        target_embedding = np.array(target_embedding)
        target_embedding = torch.FloatTensor(target_embedding).to(device)
        affinity = torch.FloatTensor(affinity).to(device)
        return drug_graph, drug_embedding, target_graph, target_embedding, affinity



