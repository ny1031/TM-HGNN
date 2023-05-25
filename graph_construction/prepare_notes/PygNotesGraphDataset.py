from torch_geometric.data import InMemoryDataset
import os.path as osp
import torch
from ConstructDatasetByNotes import *
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool as gap
import pandas as pd

IMDB_PATH = './data/IMDB_HCUT'  # path to save output hypergraphs
PRE_PATH = './data/DATA_PRE'
RAW_PATH = './data/DATA_RAW'


class PygNotesGraphDataset(InMemoryDataset):
    def __init__(self, name, split, tokenizer, dictionary, data_type='hyper', transform=None, pre_transform=None):
        self.imdb_path = osp.join(IMDB_PATH, name)
        self.name = name           # in-hospital-mortality
        self.split = split         # train / test
        self.tokenizer= tokenizer  
        self.dictionary = dictionary     
        self.data_type = data_type 
        self.pre_path = osp.join(PRE_PATH) 
        super(PygNotesGraphDataset, self).__init__(self.imdb_path, transform, pre_transform)  
        self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return osp.join(self.imdb_path)

    @property
    def processed_file_names(self):
        if self.data_type == 'hyper':
            return f'{self.split}_{self.data_type}/{self.split}_{self.data_type}.pt'
        else:
            return f'{self.split}.pt'

    def process(self):
        # construct graph by note list.
        cdbn = ConstructDatasetByNotes(pre_path=self.pre_path, split=self.split, dictionary=self.dictionary, task=self.name) 
        if self.data_type == 'hyper':
            print("Data Type : hyper")
            data_list = cdbn.construct_hypergraph_datalist()
        else:
            ValueError('error type, must be one of {cooc, hyper}...')
        
        print('\n'+'<Collate Data List...>')
        data, slices = self.collate(data_list)

        print('\n'+'<Collate Done, Start Saving...>')       
        torch.save((data, slices), osp.join(self.processed_dir, self.processed_file_names))   # '/IMDB/in-hospital-mortality' + 'train.pt'
        print('Saving Done')
        print('<<Created', self.split, 'Cutoff HyperGraph Dataset from Note List>>')


class Load_PygNotesGraphDataset(InMemoryDataset):
    def __init__(self, name, split, tokenizer, dictionary, data_type='hyper', transform=None, pre_transform=None):
        self.imdb_path = osp.join(IMDB_PATH, name)
        self.name = name           # in-hospital-mortality
        self.split = split         # train / test
        self.tokenizer= tokenizer  
        self.dictionary = dictionary     
        self.data_type = data_type 
        self.pre_path = osp.join(PRE_PATH) 
        super(PygNotesGraphDataset, self).__init__(self.imdb_path, transform, pre_transform)  
        self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return osp.join(self.imdb_path)

    @property
    def processed_file_names(self):
        if self.data_type == 'hyper':
            return f'{self.split}_{self.data_type}/{self.split}_{self.data_type}.pt'
        else:
            return f'{self.split}.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('--name', type=str, default='in-hospital-mortality')
    parser.add_argument('--pre_path', type=str, default=PRE_PATH)
    parser.add_argument('--split', type=str, default='train')  
    parser.add_argument('--action', type=str, default='create')

    args, _ = parser.parse_known_args()
    
    # vocabs generated from benchmark codes
    dictionary = open(os.path.join('/vocab.txt')).read().split()

    if args.action == 'create':
        PygNotesGraphDataset(name='in-hospital-mortality', split=args.split, tokenizer='word2vec', dictionary=dictionary)
    else:
        NotImplementedError('error action!')

