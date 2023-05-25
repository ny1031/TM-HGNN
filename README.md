# TM-HGNN

[ACL 2023] Clinical Note Owns its Hierarchy: Multi-Level Hypergraph Neural Networks for Patient-Level Representation Learning
![The proposed framework](img/tmhgnn_overview.png)

## Requirements

- CUDA=11.2
- cuDNN=8.2.0
- python=3.10.4
- pandas=1.5.3
- 

## Usage
We follow [MIMIC-III Benchmark (Harutyunyan et al.)](https://www.nature.com/articles/s41597-019-0103-9) for preprocess clinical notes.
The preprocessed NOTEEVENTS data for <code>in-hospital-mortality</code> should be in <code>data/DATA_RAW/in-hospital-mortality</code>, divided into two folders (<code>train_note</code> and <code>test_note</code>).

### Setup
```bash
pip install -r requirements.txt
```

### Prepare Notes 
```bash 
python -m graph_construction.prepare_notes.extract_cleaned_notes
python -m graph_construction.prepare_notes.create_hyper_df
```
<code>extract_cleaned_notes.py</code> cleans clinical notes in <code>data/DATA_RAW/in-hospital-mortality</code>, which results in column "Fixed TEXT" in each csv file. Word2vec token embeddings with 100 dimensions are created and saved in <code>data/DATA_RAW/root/word2vec_100</code>.

<code>create_hyper_df.py</code> creates dataframe from <code>data/DATA_RAW/in-hospital-mortality</code> where each row represents each word. The results are stored in <code>data/DATA_PRE/in-hospital-mortality</code>, divided into two folders (<code>train_hyper</code> and <code>test_hyper</code>).

### Construct Multi-level Hypergraphs
```bash
python -m graph_construction.prepare_notes.PygNotesGraphDataset
```
<code>PygNotesGraphDataset.py</code> creates multi-level hypergraphs with cutoff in <code>data/IMDB_HCUT/in-hospital-mortality</code>. 


### Model Train
```bash
python -m tmhgnn.train
```
