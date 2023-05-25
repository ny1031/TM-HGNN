import os
import pandas as pd
import re
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
import argparse
from collections import Counter
from gensim.models import Word2Vec

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)

def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))

def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)

def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section

def clean_text(text):
    """
    Clean text
    """
    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text

stemmer = WordNetLemmatizer()

def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            sent = re.sub(r'\[\*\*(.*?)\*\*\]|[_\,\d\*:~=\.\-\+\\/\"\'^&]+', ' ', sent) # re clean! added by enhwa.
            text = ' '.join([stemmer.lemmatize(word) for word in word_tokenize(
                sent
            )])
            yield text.lower()

def getText(t):
    return "\n".join(list(preprocess_mimic(t)))

def clean_docs():
    print('1. Creating vocab on whole corpus...')
    word_freq = Counter()
    for split in ['test', 'train']:
        partition = split+'_note'
        patients = list(filter(lambda x: x.find("episode") != -1, os.listdir(os.path.join(args.raw_path, args.task, partition))))
        for patient in tqdm(patients[:], desc='Iterating over patients in {}_{}'.format(args.task, partition)):
            p_df = pd.read_csv(os.path.join(args.raw_path, args.task, partition, patient), sep=',', header=0)
            notes = p_df['TEXT'].tolist()
            for note in notes:
                fixed_note = getText(note)
                for word in fixed_note.split(' '):
                    word_freq[word] += 1
    highbar = word_freq.most_common(args.most_common)[-1][1]

    print('\t\t Deciding to filter word freq in [{} ~ {}]!!'.format(args.min_freq, highbar))

    print('2. Cleaning the notes and save it on a new column "{}"'.format(args.clean_column))
    all_sents = []
    for split in ['test', 'train']:
        print("Cleaning",split,"notes...")
        partition = split + '_note'
        patients = list(filter(lambda x: x.find("episode") != -1, os.listdir(os.path.join(args.raw_path, args.task, partition))))
        for patient in tqdm(patients[:], desc='Iterating over patients in {}_{}'.format(args.task, partition)):
            p_df = pd.read_csv(os.path.join(args.raw_path, args.task, partition, patient), sep=',', header=0)
            notes = p_df['TEXT'].tolist()
            notes_ = []
            for note in notes:
                note_ = []
                sentences = getText(note).split('\n')
                for sent in sentences:
                    sent_ = []
                    for word in sent.split(' '):
                        if word_freq[word] >= args.min_freq and word_freq[word] < highbar:
                            sent_.append(word)
                            all_sents.append(sent_)
                    note_.append(' '.join(sent_))
                notes_.append('\n'.join(note_))
            p_df['fixed TEXT'] = notes_
            p_df.to_csv(os.path.join(args.raw_path, args.task, partition, patient), sep=',', index=False)

    print('3. Training {} tokenizer and save vocab in {}; save embedding in {}...'.format(args.tokenizer, args.vocab_output_path, args.token_embedding_path))
    model = Word2Vec(vector_size=int(args.dimension))
    model.build_vocab(all_sents)
    model.train(all_sents, total_examples=model.corpus_count, epochs=1)
    print('saving word2vec embedding model and vocab...')
    model.save(args.token_embedding_path)
    vocab = '\n'.join(list(model.wv.key_to_index.keys()))
    f = open(args.vocab_output_path, 'w')
    f.write(vocab)
    f.close()

    print('Complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract cleaned notes.")
    parser.add_argument('--raw_path', type=str, default='./data/DATA_RAW',
                        help="Directory where the created data should be stored.")
    parser.add_argument('--task', type=str, default='in-hospital-mortality' , help='task name: [in-hospital-mortality]')
    parser.add_argument('--tokenizer', type=str, default='word2vec')
    parser.add_argument('--dimension', type=str, default='100' , help='input dimension')
    parser.add_argument('--most_common', type=int, default=10)
    parser.add_argument('--min_freq', type=int, default=10)

    args, _ = parser.parse_known_args()
    args.vocab_output_path = os.path.join(args.raw_path, 'root', 'vocab.txt')
    args.token_embedding_path = os.path.join(args.raw_path, 'root', '{}_{}'.format(args.tokenizer, args.dimension))
    args.clean_column = 'fixed TEXT'

    clean_docs()