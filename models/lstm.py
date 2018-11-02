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
        proj = self.projection(padded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(proj, lens, batch_first=True)
        _, (hT, _) = self.lstm(packed)

        sentence_vec = hT.transpose(0, 1)  # get batch first: B*num_layers*num_directions*300
        sentence_vec = sentence_vec.contiguous().view(sentence_vec.shape[0], -1)
        return sentence_vec

    def forward(self, padded1, lens1, padded2, lens2):
        sentence_vec1 = self._sentence_features(padded1, lens1)
        sentence_vec2 = self._sentence_features(padded2, lens2)

        features = torch.cat(
            [sentence_vec1, sentence_vec1, sentence_vec1 - sentence_vec2, sentence_vec1 * sentence_vec2],
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

