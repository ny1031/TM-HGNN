from __future__ import absolute_import
from __future__ import print_function
import os
import argparse
import pandas as pd
from tqdm import tqdm

def note_hyper(p_df):
    dfs = []
    note_id = 0
    for i, note in enumerate(p_df['fixed TEXT']):
        hours = p_df.iloc[i, :]['Hours']
        category = p_df.iloc[i, :]['CATEGORY']
        hadm_id = p_df.iloc[i, :]['HADM_ID']
        subject_id = p_df.iloc[i, :]['SUBJECT_ID']
        sents = str(note).split('\n')
        sent_id = 0
        for sent in sents:
            item_list = sent.split(' ')
            for item in item_list:
                dfs.append([hours, hadm_id, subject_id, item, sent_id, note_id, category])
            sent_id += 1
        note_id += 1

    dfs = pd.DataFrame(dfs, columns=['Hours', 'HADM_ID', 'SUBJECT_ID', 'WORD', 'SENT', 'note_id', 'CATEGORY'])
    if len(dfs) == 0:
        dfs = ''
    return dfs

def create_hyper_df(partition, action):
    output_dir = os.path.join(args.pre_path, args.task, partition+'_hyper')
    split = partition + '_note'
    if action == 'make':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        patients = list(filter(lambda x: x.find("episode") != -1, os.listdir(os.path.join(args.raw_path, args.task, split))))
        for patient in tqdm(patients[:], desc='Iterating over patients in {}_{}_{}'.format(args.raw_path, args.task, split)):
            p_df = pd.read_csv(os.path.join(args.raw_path, args.task, split, patient), sep=',', header=0)
            p_hyper_df = note_hyper(p_df)
            if len(p_hyper_df) > 0:
                p_hyper_df.to_csv(os.path.join(output_dir, patient), sep='\t', index=False)
            else:
                print('--> Warning: No nodes in patient {}!!'.format(patient))

        print('training sample complete! please check in {}'.format(output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create hypergraph dataframe for in hospital mortality prediction task.")
    parser.add_argument('--raw_path', type=str, default='./data/DATA_RAW',
                        help="Directory where the cleaned data should be stored (Input).")
    parser.add_argument('--pre_path', type=str, default='./data/DATA_PRE',
                        help="Directory where the processed data should be stored (Output).")    
    parser.add_argument('--dimension', type=str, default='100' , help='input dimension')
    parser.add_argument('--task', type=str, default='in-hospital-mortality' , help='task name: [in-hospital-mortality]')
    parser.add_argument('--tokenizer', type=str, default='word2vec')
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--partition', type=str, default='test')
    parser.add_argument('--action', type=str, default='make')

    args, _ = parser.parse_known_args()

    print('Creating train HyperSamples...')
    create_hyper_df('train', args.action)
    print('Creating test HyperSamples...')
    create_hyper_df('test', args.action)
