import sys, os
sys.path.append('./prepare_notes/')
from PygNotesGraphDataset import Load_PygNotesGraphDataset as PNGD
from embedding_utils import *
from torch_geometric.data import DataLoader
import torch
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm

class LoadPaitentData():
    def __init__(self, name, type, data_type, tokenizer=''):
        self.name = name
        self.type = type
        self.data_type = data_type
        self.dictionary = open(os.path.join('vocab.txt')).read().split()
        self.tokenizer = tokenizer

        super(LoadPaitentData, self).__init__()
        self.train_set = []
        self.val_set = []
        self.test_set = []

    def split_train_val_data(self, seed, ratio, tr_split):
        np.random.seed(seed)

        cs = pd.DataFrame(self.train_set.data.y_p.tolist())[0].value_counts().to_dict()   
        train_id = np.arange(sum(cs.values())).tolist()
        print("len train_id:",len(train_id))
        print("len train_set:",len(self.train_set)) 

        np.random.shuffle(train_id)
        n_train = int(np.round(len(train_id)*(ratio)))

        self.val_set = self.train_set[train_id[n_train:]]
        self.train_set = self.train_set[train_id[:n_train]]
        return self.train_set, self.val_set

    def get_train_test(self, batch_size, seed, ratio=0.8, tr_split=1):

        print('load test data...')
        self.test_set = PNGD(name=self.name, split='test', tokenizer=self.tokenizer, dictionary=self.dictionary, data_type=self.data_type, transform=Handle_data(self.type, self.tokenizer))
        x = [data for data in self.test_set]
        print('load train data...')
        self.train_set = PNGD(name=self.name, split='train', tokenizer=self.tokenizer, dictionary=self.dictionary, data_type=self.data_type, transform=Handle_data(self.type, self.tokenizer))
        print('Train Dataset: {}'.format(len(self.train_set)))

        self.train_set, self.val_set = self.split_train_val_data(seed, ratio, tr_split)
        
        assert self.val_set.data.y_p.unique().size(0) == \
               self.train_set.data.y_p.unique().size(0) == \
               self.test_set.data.y_p.unique().size(0)


        print('Train Dataset: {}'.format(len(self.train_set)))
        print('Val Dataset: {}'.format(len(self.val_set)))
        print('Test Dataset: {}'.format(len(self.test_set)))

        num_class = self.val_set.data.y_p.unique().size(0)

        follow_batch = ['x']

        train_loader = DataLoader(self.train_set[:], batch_size=batch_size, follow_batch=follow_batch, shuffle=True)
        val_loader = DataLoader(self.val_set[:], batch_size=10, follow_batch=follow_batch, shuffle=True)
        test_loader = DataLoader(self.test_set[:], batch_size=1, follow_batch=follow_batch, shuffle=False)

        return train_loader, val_loader, test_loader, num_class


def label_distribution(data_set):
    y_1 = 0
    y_0 = 0
    for d in data_set:
        if d.y_p == torch.tensor([1]):
            y_1 += 1
        else:
            y_0 += 1
    print('#y_1: ', str(y_1), ';#y_0: ', str(y_0))
    return y_1 + y_0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GNN for EHR")
    parser.add_argument('--name', type=str, default='in-hospital-mortality')
    parser.add_argument('--type', type=str, default='cutoff')
    parser.add_argument('--data_type', type=str, default='hyper')
    parser.add_argument('--tokenizer', type=str, default='word2vec')

    args, _ = parser.parse_known_args()
    loader = LoadPaitentData(name=args.name, type=args.type, tokenizer=args.tokenizer, data_type=args.data_type)
    train_loader, val_loader, test_loader, n_class = loader.get_train_test(batch_size=1, seed=2)