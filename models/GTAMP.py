import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from models import Drug_GNN, Target_GNN
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if torch.cuda.is_available():
    device = torch.device('cuda')


class GTAMP_DTA(nn.Module):
    def __init__(self, drug_dim=64, target_dim=64, gnn_layer=10, gnn_head=4, out_dim=1):
        super(GTAMP_DTA, self).__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.n_layers = gnn_layer
        self.n_heads = gnn_head
        self.conv = 40
        self.drug_MAX_LENGH = self.drug_dim*2
        self.protein_MAX_LENGH = self.target_dim*2

        self.drug_gnn = Drug_GNN.GraphTransformer(device, n_layers=gnn_layer, node_dim=44, edge_dim=10, hidden_dim=drug_dim,
                                                        out_dim=drug_dim, n_heads=gnn_head, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)
        self.target_gnn = Target_GNN.GraphTransformer(device, n_layers=gnn_layer, node_dim=41, edge_dim=5, hidden_dim=target_dim,
                                                       out_dim=target_dim, n_heads=gnn_head, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)


        self.drug_embedding_fc = nn.Linear(300, self.drug_dim)
        self.target_embedding_fc = nn.Linear(320, self.target_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.Drug_mean_pool = nn.AvgPool1d(2, 2)
        self.Protein_mean_pool = nn.AvgPool1d(2, 2)
        
        self.modal_fc_c = nn.Linear(drug_dim * 2, self.conv*4)
        self.modal_fc_p = nn.Linear(target_dim*2,  self.conv*4)

        self.protein_attention_layer = nn.Linear(self.conv * 4, self.target_dim*2)
        self.drug_attention_layer = nn.Linear(self.conv * 4,  self.drug_dim*2)
        self.attention_layer = nn.Linear(self.drug_dim*2, self.target_dim*2)
            
        self.classifier = nn.Sequential(
            nn.Linear(748, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )


    def forward(self, drug_graph, drug_embedding, target_graph,  target_embedding, target_ids, drug_ids):

        compound_feat = self.drug_gnn(drug_graph)
        protein_feat = self.target_gnn(target_graph)

        drug_embedding = self.drug_embedding_fc(drug_embedding)
        target_embedding = self.target_embedding_fc(target_embedding)

        
        compound_feats_1 = torch.cat((drug_embedding, compound_feat), dim=1)
        compound_feats = self.modal_fc_c(compound_feats_1)
        
        protein_feats_1 = torch.cat((target_embedding, protein_feat), dim=1)
        protein_feats = self.modal_fc_p(protein_feats_1)

        drug_att = self.drug_attention_layer(compound_feats)
        protein_att = self.protein_attention_layer(protein_feats)
        
        d_att_layers = torch.unsqueeze(drug_att, 1)
        p_att_layers = torch.unsqueeze(protein_att, 2)
        
        d_att_layers = torch.unsqueeze(drug_att, 1).repeat(1, protein_att.shape[-1], 1)
        p_att_layers = torch.unsqueeze(protein_att, 2).repeat(1,1, drug_att.shape[-1])
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))

        Compound_atte = torch.mean(Atten_matrix, 1)
        Protein_atte = torch.mean(Atten_matrix, 2)
        Compound_atte = self.sigmoid(Compound_atte)
        Protein_atte = self.sigmoid(Protein_atte)
        
        compound_feats = compound_feats_1 * 0.5 + compound_feats_1 * Compound_atte
        protein_feats = protein_feats_1 * 0.5 + protein_feats_1 * Protein_atte
        
        compound_feats = compound_feats.unsqueeze(1)
        protein_feats = protein_feats.unsqueeze(1)
        
        compound_feats = self.Drug_mean_pool(compound_feats)
        protein_feats = self.Protein_mean_pool(protein_feats)
        
        compound_feats = compound_feats.squeeze(1)
        protein_feats = protein_feats.squeeze(1)
 
        num_profeats = len(protein_feats)
        esm_feats = []
        for id in target_ids:
            esm_feat = torch.from_numpy(
                np.load('/2111041015/GTAMP-DTA/data/Davis/processed/ESM_embedding/' + str(id) + '.npy',
                        allow_pickle=True))
            esm_feats.append(esm_feat)
        esm_feats = torch.stack(esm_feats).to(device)

        len_esm = esm_feats.size(0)
        if len_esm < num_profeats:
            padding_size = num_profeats - len_esm
            esm_feats = F.pad(esm_feats, (0, 0, 0, padding_size), 'constant', 0)
        elif len_esm > num_profeats:
            esm_feats = esm_feats[:num_profeats, :]

        protein_feats = torch.cat((protein_feats, esm_feats), dim=1)

        num_compound_feats = len(compound_feats)
        mol2vec_feats = []

        for id in drug_ids:
          mol2vec_feat = torch.from_numpy(
              np.load('/2111041015/GTAMP-DTA/data/Davis/processed/mol2vec_embedding1/' + str(id) + '.npy',
                        allow_pickle=True))
          mol2vec_feats.append(mol2vec_feat)

        mol2vec_feats = torch.stack(mol2vec_feats).to(device)


        len_mol2vec = mol2vec_feats.size(0)
        if len_mol2vec < num_compound_feats:
            padding_size = num_compound_feats - len_mol2vec
            mol2vec_feats = F.pad(mol2vec_feats, (0, 0, 0, padding_size), 'constant', 0)
        elif len_mol2vec > num_compound_feats:
            mol2vec_feats = mol2vec_feats[:num_compound_feats, :]

        compound_feats = torch.cat((compound_feats, mol2vec_feats), dim=1)
        
        dp = torch.cat((compound_feats, protein_feats), dim=1)
        dp = dp.to(torch.float32)
        x = self.classifier(dp)

        return x


