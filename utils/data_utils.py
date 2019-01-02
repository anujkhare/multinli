import json

import torch
from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import pandas as pd
import string


class MNLIDataset(Dataset):
    @staticmethod
    def _sort_by_len(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()  # type: pd.DataFrame
        df['len1'] = df.sentence1.apply(lambda x: len(x.split()))
        df['len2'] = df.sentence2.apply(lambda x: len(x.split()))
        df['total_len'] = df.len1 + df.len2

        df.sort_values(by='total_len', ascending=True, inplace=True)
        return df

    def __init__(self, df: pd.DataFrame, word_vectors: Dict, sort_by_len: bool = True) -> None:
        df['label_id'] = df.gold_label.map(lbl_to_id)
        assert df.gold_label.notnull().all()
        df['sentence1'] = df['sentence1'].apply(lambda x: x.strip(string.punctuation))
        df['sentence2'] = df['sentence2'].apply(lambda x: x.strip(string.punctuation))

        if sort_by_len:
            df = self._sort_by_len(df)
        self.df = df
        self.word_vectors = word_vectors

    def __len__(self) -> int:
        return len(self.df)

    def _sentence_to_vec(self, sentence: str) -> np.ndarray:
        # FIXME
        vectors = []
        final_sentence = []
        for word in sentence.split(' '):
            if word not in self.word_vectors:
                vectors.append(np.zeros(300, dtype=np.float32))
                continue

            final_sentence.append(word)
            vectors.append(self.word_vectors[word])

        #         vectors = np.vstack(vectors)
        vectors = np.array(vectors)
        final_sentence = ' '.join(final_sentence)
        return vectors, final_sentence

    def _preprocess(self, record):
        # Convert sentences to word vectors, return list of
        v1, fs1 = self._sentence_to_vec(record['sentence1'])
        v2, fs2 = self._sentence_to_vec(record['sentence2'])
        return {
            'sentence1': v1,
            'sentence2': v2,
            'label': record['label_id'],
            'final_sentence1': fs1,
            'final_sentence2': fs2,
        }

    def __getitem__(self, ix):
        return self._preprocess(self.df.iloc[ix])


def load_word_vectors(file_path: str) -> Dict:
    with open(file_path, 'r') as infile:
        data_glove = infile.read().split('\n')

    data_glove = map(lambda x: x.split(), data_glove)  # Split the words

    glove = {
        line[0]: np.array(line[1:], dtype=np.float32)
        for line in data_glove
        if len(line) == 301
    }

    return glove


def load_data(file_path) -> pd.DataFrame:
    with open(file_path, 'r') as infile:
        data = infile.read().split('\n')

    data = list(map(json.loads, data[:-1]))

    df = pd.DataFrame(data)

    print(len(df))
    df = df.loc[df.gold_label != '-']
    print(len(df))
    return df


id_to_lbl = {
    0: 'neutral',
    1: 'entailment',
    2: 'contradiction',
    -100: 'hidden'
}

lbl_to_id = {
    lbl: id
    for id, lbl in id_to_lbl.items()
}


# Data loader specific
def get_padded_tensor_and_lens(data):
    # NOTE: DON'T SORT HERE! YOU'LL LOSE THE CORRESPONDANCE B/W SENTENCE PAIRS AND THE LABELS
    lens = [len(x) for x in data]

    max_len = max(lens)
    data = np.array([
        np.pad(datum, pad_width=[(0, max_len - len(datum)), (0, 0)], mode='constant', constant_values=0)
        for datum in data
    ])

    tensor = torch.from_numpy(data.astype(np.float32))
    return tensor, lens


def collate_fn(batch):
    sentence1 = get_padded_tensor_and_lens([sample['sentence1'] for sample in batch])
    sentence2 = get_padded_tensor_and_lens([sample['sentence2'] for sample in batch])

    final_sentence1 = [sample['final_sentence1'] for sample in batch]
    final_sentence2 = [sample['final_sentence2'] for sample in batch]
    label = torch.from_numpy(np.array([sample['label'] or -100 for sample in batch], dtype=np.long))

    return {
        'sentence1': sentence1,
        'sentence2': sentence2,
        'final_sentence1': final_sentence1,
        'final_sentence2': final_sentence2,
        'label': label
    }
