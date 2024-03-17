import math
import random
import timeit
import pandas as pd
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import argparse
# import os 

from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error, r2_score

from metrics import *
from Dataset import Dataset_DTA
from models.GTAMP import GTAMP_DTA

device = torch.device('cuda')



def train(model, device, train_loader, optimizer):

    root_file = '/2111041015/GTAMP-DTA'
    file_path = root_file + '/data/' + args.dataset + '/DTA/fold/5/' + args.dataset + '_train.csv'
    raw_data = pd.read_csv(file_path)
    pro_indexs = raw_data['PROTEIN_ID'].values


    com_indexs = raw_data['COMPOUND_ID'].values
    infeter = 0

    model.train()
    for batch_idx, data in enumerate(train_loader):
        label = data[-1].to(device)
        drug_graph, drug_embedding, target_graph, target_embedding = data[:-1]
        
        
        drug_len = drug_embedding.shape[0]
        drug_ids = com_indexs[infeter: infeter+drug_len]

        target_len = target_embedding.shape[0]
        target_ids = pro_indexs[infeter: infeter+target_len]
        infeter = infeter+target_len

        drug_graph = drug_graph.to(device)
        drug_embedding = drug_embedding.to(device)
        target_graph = target_graph.to(device)
        target_embedding = target_embedding.to(device)

        output = model(drug_graph, drug_embedding, target_graph,  target_embedding, target_ids, drug_ids)
        loss = criterion(output, label.view(-1, 1).float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    
    root_file = '/2111041015/GTAMP-DTA'
    file_path = root_file + '/data/' + args.dataset + '/DTA/fold/5/' + args.dataset + '_test.csv'
    raw_data = pd.read_csv(file_path)
    pro_indexs = raw_data['PROTEIN_ID'].values
    com_indexs = raw_data['COMPOUND_ID'].values
    infeter = 0
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            label = data[-1].to(device)
            drug_graph, drug_embedding, target_graph, target_embedding = data[:-1]
            
            drug_len = drug_embedding.shape[0]
            drug_ids = com_indexs[infeter: infeter+drug_len]
            target_len = target_embedding.shape[0]
            target_ids = pro_indexs[infeter: infeter+target_len]
            infeter = infeter+target_len

            drug_graph = drug_graph.to(device)
            drug_embedding = drug_embedding.to(device)
            target_graph = target_graph.to(device)
            target_embedding = target_embedding.to(device)
            output = model(drug_graph, drug_embedding, target_graph, target_embedding, target_ids, drug_ids)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()


    MSE = mse(total_labels, total_preds)
    Pearson = pearson(total_labels, total_preds)
    CI = ci(total_labels, total_preds)
    RM2 = rm2(total_labels, total_preds)
    return MSE, Pearson, CI, RM2, total_labels, total_preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Davis', help='dataset')
    parser.add_argument('--fold', type=str, default='5', help='fold of dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--batch', type=int, default=16, help='batchsize')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drug_dim', type=int, default=64, help='dimension of drug embedding')
    parser.add_argument('--target_dim', type=int, default=64, help='dimension of target embedding')
    parser.add_argument('--gnn_layer', type=int, default=10, help='layer of GNN')
    parser.add_argument('--gnn_head', type=int, default=4, help='head of GNN')

    args = parser.parse_args()

    args.file_path = '/2111041015/GTAMP-DTA/data/' + args.dataset + '/processed/DTA'

    train_set = Dataset_DTA(dataset=args.dataset, drug_graph=args.file_path + '/train/fold/' + args.fold +'/drug_graph.bin',
                           drug_embedding=args.file_path + '/train/fold/' + args.fold + '/drug_embedding_max.npy',
                           compound_id=args.file_path + '/train/fold/' + args.fold +'/protein_id.npy',
                           target_graph=args.file_path + '/train/fold/' + args.fold +'/drug_graph.bin',
                           target_embedding=args.file_path + '/train/fold/' + args.fold +'/protein_embedding_max.npy',
                           target_id=args.file_path + '/train/fold/' + args.fold +'/drug_id.npy',
                           affinity=args.file_path + '/train/fold/' + args.fold +'/label.npy')
    test_set = Dataset_DTA(dataset=args.dataset, drug_graph=args.file_path + '/test/fold/' + args.fold +'/drug_graph.bin',
                          drug_embedding=args.file_path + '/test/fold/' + args.fold + '/drug_embedding_max.npy',
                          compound_id=args.file_path + '/test/fold/' + args.fold +'/protein_id.npy',
                          target_graph=args.file_path + '/test/fold/' + args.fold +'/drug_graph.bin',
                          target_embedding=args.file_path + '/test/fold/' + args.fold +'/protein_embedding_max.npy',
                          target_id=args.file_path + '/test/fold/' + args.fold +'/drug_id.npy',
                          affinity=args.file_path + '/test/fold/' + args.fold +'/label.npy',)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=train_set.collate, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, collate_fn=test_set.collate, drop_last=True)

    model = GTAMP_DTA(drug_dim=args.drug_dim, target_dim=args.target_dim, gnn_layer=args.gnn_layer, gnn_head=args.gnn_head, out_dim=1)
    model.to(device)


    start = timeit.default_timer()
    best_ci = 0
    best_mse = 100
    best_pearson = 0
    best_rm2 = 0
    best_r2 = 0
    best_epoch = -1
    file_model = 'model_save/' + args.dataset + '/fold/' + args.fold + '/'

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=80, verbose=True, min_lr=1e-5)
    criterion = nn.MSELoss()

    Indexes = ('Epoch\t\tTime\t\tMSE\t\tPearson\t\tCI\t\tr2')

    """Start training."""
    print('Dataset:' + args.dataset + ', fold:' + args.fold)
    print(Indexes)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer)
        mse_test, pearson_test, ci_test, rm2_test, total_labels, total_preds = test(model, device, test_loader)
        scheduler.step(mse_test)
        scheduler.step(pearson_test)
        scheduler.step(ci_test)
        scheduler.step(rm2_test)
        end = timeit.default_timer()
        time = end - start
        ret = [epoch + 1, round(time, 2), round(mse_test, 4), round(pearson_test, 4), round(ci_test, 4), round(rm2_test, 4)]
        print('\t\t'.join(map(str, ret)))

        #best_mse
        if mse_test < best_mse:
            best_epoch = epoch + 1
            best_mse = mse_test

            np.save('./Ddata_labels6.npy', total_labels)
            np.save('./Ddata_preds6.npy', total_preds)

            str_best_mse = f"best_mse:{best_mse}"
            with open('./Ddata6.txt', 'a') as f:
                f.write(str_best_mse)
                f.write("\n")

            print('MSE improved at epoch ', best_epoch, ';\tbest_mse:', best_mse)
        
        #best_pearson   "best_mse:"
        if pearson_test > best_pearson:
            best_epoch = epoch + 1
            best_pearson = pearson_test

            str_best_pearson = f"best_pearson:{best_pearson}"
            with open('./Ddata6.txt', 'a') as f:
                f.write(str_best_pearson)
                f.write("\n")

            print('Pearson improved at epoch ', best_epoch, ';\tbest_pearson:', best_pearson)
        
        #best_ci
        if ci_test > best_ci:
            best_epoch = epoch + 1
            best_ci = ci_test

            str_best_ci = f"best_ci:{best_ci}"
            with open('./Ddata6.txt', 'a') as f:
                f.write(str_best_ci)
                f.write("\n")

            print('CI improved at epoch ', best_epoch, ';\tbest_ci:', best_ci)

        #best_rm2
        if rm2_test > best_rm2:
            best_epoch = epoch + 1
            best_rm2 = rm2_test

            str_best_rm2 = f"best_rm2:{best_rm2}"
            with open('./Ddata6.txt', 'a') as f:
                f.write(str_best_rm2)
                f.write("\n")

            print('rm2 improved at epoch ', best_epoch, ';\tbest_rm2:', best_rm2)
            
        with open('./Ddata6.txt', 'a') as f:
            f.write("\n")
