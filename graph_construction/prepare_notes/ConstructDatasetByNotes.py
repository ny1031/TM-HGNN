import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
import torch, sys
from torch_geometric.data import Data
pd.set_option('display.max_columns', None)
import os.path as osp
import os
from tqdm import tqdm
from gensim.models import Word2Vec

def graph_to_torch_sparse_tensor(G_true, node_attr=None):
    G = nx.convert_node_labels_to_integers(G_true)
    A_G = np.array(nx.adjacency_matrix(G, weight='edge_type').todense())
    # """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse.csr_matrix(A_G).tocoo()
    edge_index = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col))).to(torch.long)
    edge_attrs = torch.from_numpy(sparse_mx.data).to(torch.float32)

    x = []
    batch_n = []
    batch_t = []
    for node in range(len(G)):
        x.append(G.nodes[node]['node_emb'])
        if node_attr != None:
            for attr in node_attr:
                if attr == 'note_id':
                    batch_n.append(G.nodes[node][attr])
                elif attr == 'cat_id':
                    batch_t.append(G.nodes[node][attr])

    x = torch.from_numpy(np.array(x)).to(torch.float32)
    batch_n = torch.from_numpy(np.array(batch_n)).to(torch.long)
    batch_t = torch.from_numpy(np.array(batch_t)).to(torch.long)

    return edge_index, edge_attrs, x, batch_t, batch_n

def generate_patient_graph(df):
    # print('\nraw---df: ', len(df))
    result_df = combine_same_word_pair(df, col_name='global_freq')   
    result_df['edge_attr'] = 1                                         
    result_graph = nx.from_pandas_edgelist(result_df, 'word1', 'word2', 'edge_attr')
 
    ### remove nan nodes ###
    remove_list = []
    for node in result_graph:
        if node != node:
            remove_list.append(node)
           # result_graph.remove_node(node)
        elif str(node) == 'nan':
            remove_list.append(node)
           # result_graph.remove_node(node)
        else:
            continue
            
    if len(remove_list) > 0 :
        for rm_node in remove_list:
            result_graph.remove_node(rm_node)
            
    return result_graph

def load_word2vec_embeddings():
    # word2vec pretrained embeddings
    w2v_path = './data/DATA_RAW/root/word2vec_100'
    print('import pretrained word representation from {}...'.format(w2v_path))
    word_embeddings = Word2Vec.load(osp.join(w2v_path)).wv
    return word_embeddings

class ConstructDatasetByNotes():
    def __init__(self, pre_path, split, dictionary, task):
        self.pre_path = pre_path
        self.split = split
        self.dictionary = dictionary
        self.task = task        
        super(ConstructDatasetByNotes).__init__()
        self.labels = self.get_labels(split)          
        self.cat_path = osp.join(self.pre_path, 'categories.txt')
        # self.all_cats = [a.strip() for a in open(self.cat_path).readlines()]   # Nutrition ~ Respiratory
        
        '''
        categories.txt: (total 14)
       
        Nutrition
        ECG
        Rehab Services
        Case Management 
        Echo
        Pharmacy
        Physician 
        Nursing
        Consult
        General
        Nursing/other
        Radiology
        Social Work
        Discharge summary
        Respiratory 
        '''
        
    def get_labels(self, split):
        label_patients = pd.read_csv(osp.join(self.pre_path, self.task, split+'_hyper', 'listfile.csv'), sep=',', header=0)
        label_patients['name'] = label_patients.apply(lambda x: str(x['patient'])+'_'+x['episode'], axis=1)    
        label_patients = label_patients.loc[:, ['name', 'y_true']]
        return label_patients          

    def make_embedding(self, G, node, node_type):
        emb = np.zeros(104)
        '''
        0: node_type -> {0:word, 1:note; 2:taxonomy}
        1: word_id 
        2: note_id 
        3: taxonomy_id 
        4:~: for embedding. (word embedding initialized by word2vec)
        '''
        if node_type == 'word':
            emb[0] = 0
            emb[1] = -1
            emb[2] = G.nodes[node]['note_id']
            emb[3] = G.nodes[node]['cat_id']
        elif node_type == 'note':
            emb[0] = 1
            emb[1] = -1
            emb[2] = G.nodes[node]['note_id']
            emb[3] = G.nodes[node]['cat_id']
        elif node_type == 'tax':
            emb[0] = 2
            emb[1] = -1
            emb[2] = -1
            emb[3] = G.nodes[node]['cat_id']
        return emb
    
    def create_all_cats(self, path):
        all_cats = []
        for split in ['train', 'test']:
            hyper_path = osp.join(self.pre_path, self.task, split+'_hyper')
            patients = list(filter(lambda x: x in os.listdir(hyper_path), list(self.labels['name'])))  
            for patient in tqdm(patients[:], desc='Iterating over patients in {}_hyper'.format(split)): 
                p_df = pd.read_csv(osp.join(hyper_path, patient), sep='\t', header=0)
                all_cats += p_df['CATEGORY'].tolist()
        all_cats = list(set(all_cats))
        f = open(f'{path}/categories.txt', 'w')
        f.write('\n'.join(all_cats))
        f.close()

    def set_node_embedding(self, G, node_attr='node_emb', word_embeddings=None):
        for node in G:
            node = str(node)
            if node_attr=='node_emb':
                if len(word_embeddings.index_to_key) > 0:  # default tokenizer
                    if node in self.dictionary:
                        emb = self.make_embedding(G, node, node_type='word')
                        emb[1] = np.array([word_embeddings.index_to_key.index(node)])
                        emb[4:] = np.array(word_embeddings.get_vector(node))
                    elif 'n_' in node:
                        emb = self.make_embedding(G, node, node_type='note')
                    else:
                        print(node)
                        raise ValueError('error assignment')
                else:  
                    emb = np.array([node])
            elif node_attr == 'pe':
                emb = node_attr[node_attr[:, 0] == node, 1:][0]
                assert (emb.astype(np.float32) == 1).sum() > 0
            else:
                raise ValueError('unknown node attribute')

            G.nodes[node][node_attr] = emb
            
        return G

### HyperGraph ###
    def construct_hypergraph_datalist(self):
        print()
        print("<<Start Construct Hypergraph Datalist>>")
        word_embeddings = load_word2vec_embeddings()

        hyper_path = osp.join(self.pre_path + '/' + self.task + '/', self.split + '_hyper/')
        patients = list(filter(lambda x: x in os.listdir(hyper_path), list(
            self.labels['name'])))  
        list_all_cats = [a.strip() for a in open(self.cat_path).readlines()]   # Nutrition ~ Respiratory
        # episode file names into list
        print('<Patient list generation done>')
        Data_list = []
        for patient in tqdm(patients[:], desc='Iterating over patients in {}_hyper'.format(self.split)):
            p_df = pd.read_csv(osp.join(hyper_path, patient), sep='\t', header=0) 
            # col : ['Hours', 'HADM_ID', 'SUBJECT_ID', 'WORD', 'SENT', 'note_id', 'CATEGORY']

            # change into str if word is not str (int or else...)
            # drop NaN
            p_df = p_df.dropna(axis=0)

            ### Use only 6 major categories ###
            p_df.loc[:, 'CATEGORY'] = p_df.CATEGORY.apply(lambda x: x.strip())  

            # if no notes for 6 CATEGORY => continue (skip to next episode)
            if len(p_df[p_df['CATEGORY'].isin(
                    ['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician'])]) == 0:
                continue

            # filter only 6 categories
            p_df = p_df[p_df['CATEGORY'].isin(['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician'])]

            ### Cutoff notes < 30 ###
            if p_df['note_id'].nunique() > 30:
                max_nid = p_df['note_id'].unique()[30]  # 30 th note_id
                p_df = p_df[p_df['note_id'] < max_nid]  # cut off 0~29 (max 30 notes)

            p_df['WORD'] = p_df['WORD'].astype(str)
            p_df['SENT'] = p_df['SENT'].astype(str)
            p_df['note_id'] = p_df['note_id'].astype(str)
            p_df['note_NM'] = "n_" + p_df['note_id']

            # p_df['SENT_NM'] = p_df['SENT'] + '_n_' + p_df['note_id'] + '_' + p_df['CATEGORY']

            y_p = self.labels[self.labels['name'] == patient]['y_true'].values[0]
            y_p = torch.from_numpy(np.array([y_p])).to(torch.long)  # tensor([1]) OR tensor([0])
            G_n_list = []
            y_n_list = []
            hour_n_list = []

            # per note id generate G_n
            for n_id, n_df in p_df.groupby(by='note_NM'):

                # drop NaN
                n_df = n_df.dropna(axis=0)
                n_df = n_df[n_df['WORD'] != 'nan']

                if len(n_df) > 0:
                    ## edge_type == 1 for word-note edge
                    n_df['edge_type'] = 1 # word-note_edge_type = 0
                    
                    # Graph per note
                    G_n = nx.from_pandas_edgelist(n_df, 'WORD', 'note_NM', 'edge_type')
                        
                    ### Cutoff words per note < 300 ###
                    cut_300 = list(G_n.nodes)[300:]
                    G_n.remove_nodes_from(cut_300)

                    # note ids to node attributes
                    cat_id = list_all_cats.index(n_df['CATEGORY'].values[0].rstrip())  # 14 categories 
                    note_id = int(n_df['note_id'].values[0])
                    attrs = {}
                    for node in G_n:
                        attrs[node] = {'note_id': note_id, 'cat_id': cat_id}
                    nx.set_node_attributes(G_n, attrs)

                    # set node embeddings
                    G_n = self.set_node_embedding(G_n, node_attr='node_emb', word_embeddings=word_embeddings)
                    G_n_list.append(G_n)

                    y_n_list.append([cat_id])
                    hour_n_list.append([n_df['Hours'].values[0]])
                else:
                    continue
            G_n = nx.disjoint_union_all(G_n_list)
            
            for node in range(len(G_n)):
                if G_n.nodes[node]['node_emb'][0] == 0:
                    tax_node = 't_'+ str(G_n.nodes[node]['cat_id'])
                    
                    ## edge_type == 2 for word-taxonomy edge
                    G_n.add_edge(node, tax_node, edge_type=2) # word-taxonomy_edge_type = 1
                    if 'node_emb' not in G_n.nodes[tax_node]:
                        emb = self.make_embedding(G_n, node, node_type='tax')
                        G_n.nodes[tax_node]['node_emb'] = emb # for embedding.
                    if 'note_id' not in G_n.nodes[tax_node]:
                        G_n.nodes[tax_node]['note_id'] = G_n.nodes[node]['note_id']
                        G_n.nodes[tax_node]['cat_id'] = G_n.nodes[node]['cat_id']
                        

            edge_index_n, edge_index_mask, x_n, batch_t, batch_n = graph_to_torch_sparse_tensor(G_n, node_attr=['note_id', 'cat_id'])

            y_n = torch.from_numpy(np.array(y_n_list)).to(torch.long)
            hour_n = torch.from_numpy(np.array(hour_n_list)).to(torch.float32)


            data = Data(x_n=x_n, edge_index_n=edge_index_n, edge_index_mask=edge_index_mask, hour_n=hour_n, y_n=y_n, y_p=y_p, batch_n=batch_n, batch_t=batch_t)
            Data_list.append(data)

        print('<Hypergraph Data list generation done>')
        print('<<End Construct Hypergraph Datalist>>')
        print()

        return Data_list


if __name__ == '__main__':
    task = 'in-hospital-mortality'
    raw_path = '/DATA_RAW/'
    pre_path = '/DATA_PRE/{}'.format(task)
    dictionary = open(os.path.join('/', 'vocab.txt')).read().split()
    cdbn = ConstructDatasetByNotes(pre_path, split='train', dictionary=dictionary)  # split = train. test
    data_list = cdbn.construct_hypergraph_datalist()