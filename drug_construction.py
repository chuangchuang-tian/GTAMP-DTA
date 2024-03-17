import dgl
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from dgl.data.utils import save_graphs

from scipy import sparse as sp
from itertools import permutations
from scipy.spatial import distance_matrix
from gensim.models import word2vec
from mol2vec.features import MolSentence, mol2alt_sentence, mol2sentence, sentences2vec
from dgl import load_graphs

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore")

ATOM_SET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def smiles_embedding(line, set, max_length=100):
    X = np.zeros(max_length, dtype=np.int64())
    for i, ch in enumerate(line[:max_length]):
        X[i] = set[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


def node_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return results


def edge_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    g = dgl.DGLGraph()
    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    node_feature = np.array([node_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        node_feature = np.concatenate([node_feature, chiral_arr], axis=1)

    g.ndata["atom"] = torch.tensor(node_feature)

    # Add edges
    src_list = []
    dst_list = []
    edge_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edge_feature = edge_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        edge_feats_all.append(edge_feature)
        edge_feats_all.append(edge_feature)

    g.add_edges(src_list, dst_list)

    g.edata["bond"] = torch.tensor(np.array(edge_feats_all))
    g = laplacian_positional_encoding(g, pos_enc_dim=8)
    return g

if __name__ == '__main__':
    dataset = 'Davis'
    fold = 5
    #/2111041015/GTAMP-DTA
    
    file_path = '/2111041015/GTAMP-DTA/data/' + dataset + '/DTA/fold/'
    # file_path_id = 'data/' + dataset + '/DTA/' + dataset + '_drugID_unique.csv'
    file_path_drug = '/2111041015/GTAMP-DTA/data/' + dataset + '/DTA/' + dataset + '_compound_mapping.csv'
    dir_output = ('/2111041015/GTAMP-DTA/data/' + dataset + '/processed/DTA/')
    os.makedirs(dir_output, exist_ok=True)

    train_data = pd.read_csv(file_path + str(fold)  + '/' + dataset + '_train.csv')
    test_data = pd.read_csv(file_path + str(fold)  + '/' + dataset + '_test.csv')

    # raw_data_id = pd.read_csv(file_path_id)
    raw_data_drug = pd.read_csv(file_path_drug)
    drug_values = raw_data_drug['COMPOUND_SMILES'].values
    drug_id_unique = raw_data_drug['COMPOUND_ID'].values

    drug_id_train = train_data['COMPOUND_ID'].values
    drug_id_test = test_data['COMPOUND_ID'].values

    label_train = train_data['REG_LABEL'].values
    label_test = test_data['REG_LABEL'].values


    N = len(drug_values)
    drug_max_len = 100
    drugs_id_train, drugs_id_test, drugs_id_val = list(), list(), list()
    labels_train, labels_test, labels_val = list(), list(), list()

    # drug_map = pd.read_csv('data/' + dataset + '/' + dataset + '_drug_mapping.csv')
    # drug_to_id = {}
    # id_to_drug = {}
    # for smiles, id in zip(list(drug_map['drug_SMILES']), list(drug_map['drug_ID'])):
    #     drug_to_id[smiles] = id
    # for id, smiles in zip(list(drug_map['drug_ID']), list(drug_map['drug_SMILES'])):
    #     id_to_drug[id] = smiles

    # for no, data in enumerate(drug_id_unique):
    #     drugs_g = list()
    #     print('/'.join(map(str, [no + 1, N])))
    #     # drug_id = drug_to_id[data]
    #     smiles_data = drug_values[no]
    #     drug_graph = smiles_to_graph(smiles_data)
    #     # drug_embedding = smiles_embedding(data, ATOM_SET, drug_max_len)
    #     drugs_g.append(drug_graph)
    #     dgl.save_graphs(dir_output + '/drug_graph/' + str(data) + '.bin', list(drugs_g))
        # np.save(dir_output + '/drug_embedding/' + drug_id + '.npy', list(drug_embedding))
        # drugs_e.append(drug_embedding)
    
    # print("drug_id_unique: ", drug_id_unique)
    # for no, data in enumerate(drug_id_unique):
    #     # drugs_e = list()
    #     print('/'.join(map(str, [no + 1, N])))
    #     # seq = protein_seq_unique[no]
    #     # protein_id = protein_to_id[seq]
    #     # tape_feats = sequence_to_tape(seq)
    #     drug_graph, _ = load_graphs(
    #         '/2111041015/2.8improveD52zi/data/' + dataset + '/processed/drug_graph/' + str(data) + '.bin')
    #     feats = drug_graph[0].ndata['atom'][:, 44:]
    #     np.save('/2111041015/2.8improveD52zi/data/' + dataset + '/processed/mol2vec_embedding1/' + str(data) + '.npy', feats)

    ## drug_embedding process
    drugs_embedding_train, drugs_embedding_test = [], []
    N = len(drug_id_train)
    for no, id in enumerate(drug_id_train):
        print('/'.join(map(str, [no + 1, N])))
        drug_embedding_train = np.load('/2111041015/GTAMP-DTA/data/' + dataset + '/processed' + '/mol2vec_embedding/' + str(id) + '.npy', allow_pickle=True)
        drug_embedding_train = drug_embedding_train[0]
        np.save('/2111041015/GTAMP-DTA/data/' + dataset + '/processed/mol2vec_embedding1/' + str(id) + '.npy', drug_embedding_train)
        drugs_embedding_train.append(drug_embedding_train)
    print(len(drugs_embedding_train))
    print("max:")
    np.save('/2111041015/GTAMP-DTA/data/Davis/processed/DTA/train/fold/5/drug_embedding_max.npy', drugs_embedding_train)

    N = len(drug_id_test)
    for no, id in enumerate(drug_id_test):
        print('/'.join(map(str, [no + 1, N])))
        drug_embedding_test = np.load('/2111041015/GTAMP-DTA/data/' + dataset + '/processed' + '/mol2vec_embedding/' + str(id) + '.npy', allow_pickle=True)
        drug_embedding_test = drug_embedding_test[0]
        drugs_embedding_test.append(drug_embedding_test)
    print(len(drugs_embedding_test))
    np.save('/2111041015/GTAMP-DTA/data/Davis/processed/DTA/test/fold/5/drug_embedding_max.npy', drugs_embedding_test)


    ## drug_graph process
    drugs_graph_train, drugs_graph_test = [], []
    N = len(drug_id_train)
    for no, id in enumerate(drug_id_train):
        print('/'.join(map(str, [no + 1, N])))
        drug_graph_train, _ = load_graphs('/2111041015/GTAMP-DTA/data/' + dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        drugs_graph_train.append(drug_graph_train[0])
    print(len(drugs_graph_train))
    dgl.save_graphs(dir_output + '/train/fold/' + str(fold) + '/drug_graph.bin', drugs_graph_train)

    N = len(drug_id_test)
    for no, id in enumerate(drug_id_test):
        print('/'.join(map(str, [no + 1, N])))
        drug_graph_test, _ = load_graphs('/2111041015/GTAMP-DTA/data/' + dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        drugs_graph_test.append(drug_graph_test[0])
    print(len(drugs_graph_test))
    dgl.save_graphs(dir_output + '/test/fold/' + str(fold) + '/drug_graph.bin', drugs_graph_test)
    #
    # drug_id process
    N = len(drug_id_train)
    for no, id in enumerate(drug_id_train):
        print('/'.join(map(str, [no + 1, N])))
        drugs_id_train.append(id)
    np.save(dir_output + '/train/fold/' + str(fold) + '/drug_id.npy', drugs_id_train)
    #
    N = len(drug_id_test)
    for no, id in enumerate(drug_id_test):
        print('/'.join(map(str, [no + 1, N])))
        drugs_id_test.append(id)
    np.save(dir_output + '/test/fold/' + str(fold) +'/drug_id.npy', drugs_id_test)


    ## Label process
    N = len(label_train)
    for no, data in enumerate(label_train):
        print('/'.join(map(str, [no + 1, N])))
        labels_train.append(data)
    np.save(dir_output + '/train/fold/' + str(fold) + '/label.npy', labels_train)

    N = len(label_test)
    for no, data in enumerate(label_test):
        print('/'.join(map(str, [no + 1, N])))
        labels_test.append(data)
    np.save(dir_output + '/test/fold/' + str(fold) + '/label.npy', labels_test)

    print('The preprocess of ' + dataset + ' dataset has finished!')
