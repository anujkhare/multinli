import numpy as np
import torch
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, vector_dim=300, proj_dim=300, num_layers=1, bidirectional=False, linear_size=2048) -> None:
        super().__init__()

        self.projection = torch.nn.Linear(vector_dim, proj_dim)
        self.lstm = torch.nn.LSTM(proj_dim, proj_dim, num_layers=num_layers, batch_first=True,
                                  bidirectional=bidirectional)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(proj_dim * num_layers * (1 + bidirectional) * 4, linear_size),
            torch.nn.Linear(linear_size, 3),
        )

    def _sentence_features(self, padded, lens):
        # Take projection of all word embeddings
        proj = self.projection(padded)
        
        # You need to sort and pack the sentences
        lens = np.array(lens, dtype=np.long)
        sort_inds = np.argsort(lens)[::-1]

        idx = np.arange(len(sort_inds))
        inds_reverse = [np.where(sort_inds == ix)[0][0] for ix in idx]
        assert np.all(lens[sort_inds][inds_reverse] == lens)
        
        # Acutally sort the sentences and their lengths
        padded_sorted = torch.cat([proj[ind][np.newaxis, ...] for ind in sort_inds], dim=0)
        lens = lens[sort_inds]
        assert len(padded_sorted) == len(proj)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sorted, lens, batch_first=True)
        
        # Pass through the LSTM to calculate the final hidden state
        _, (hT, _) = self.lstm(packed)
        sentence_vec = hT.transpose(0, 1)
        sentence_vec = sentence_vec.contiguous().view(sentence_vec.shape[0], -1)

        # Reorder the sentences to the initial order
        sentence_vec = sentence_vec[inds_reverse]
        
        return sentence_vec

    def forward(self, padded1, lens1, padded2, lens2):
        sentence_vec1 = self._sentence_features(padded1, lens1)
        sentence_vec2 = self._sentence_features(padded2, lens2)

        features = torch.cat(
            [sentence_vec1, sentence_vec2, sentence_vec1 - sentence_vec2, sentence_vec1 * sentence_vec2],
            dim=1)
        pred = self.classifier(features)
        return F.log_softmax(pred, dim=1)


def test_lstm_unpadded():
    bsz = 3
    seq1 = torch.from_numpy(np.random.rand(bsz, 20, 300, ).astype(np.float32))
    seq2 = torch.from_numpy(np.random.rand(bsz, 24, 300, ).astype(np.float32))
    lens1 = [20] * 3
    lens2 = [24] * 3

    model = LSTM()

    outputs = model(seq1, lens1, seq2, lens2)
    assert outputs.shape == (bsz, 3)


def test_lstm_padded(dataloader):
    model = LSTM()

    # FIXME: create the data
    batch = next(iter(dataloader))
    bsz = len(batch['label'])
    outputs = model(*batch['sentence1'], *batch['sentence2'])
    assert outputs.shape == (bsz, 3)

