from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F

from models.scaled_dot_attention import ScaledDotAttention


class LSTMAttention(torch.nn.Module):
    """
    The main intuition for this model is that while combining all the word vectors in a given sentence to form a single
    sentence vector, rather than taking the final hidden state of the LSTM, we look at the hidden context vectors over
    each word (hidden state of the LSTM) and try to take some combination of those.
    
    This combination is achieved by using attention: the vector for each word acts as the keys (and values). For the query,
    we learn a new, unique, "query" vector. Intuitively, this query vector is supposed to be able to give a higher weight
    to words that are more relevant for classification.
    """
    def __init__(self, n_out=3, vector_dim=300, proj_dim=300,
                 num_layers=1, birnn=True,
                 linear_dim = 512,
                 query_size=300, attention_dim=299, device=None,
                 query_requires_grad:bool = True, query_min=0, query_max=2,
                ) -> None:
        super().__init__()
        
        
        self.projection = torch.nn.Linear(vector_dim, proj_dim)
        self.lstm = torch.nn.LSTM(proj_dim, proj_dim, num_layers=num_layers, batch_first=True, bidirectional=birnn)
        
        hidden_size = proj_dim * num_layers * (int(birnn) + 1)

        # Attention: to combine the hidden states of the LSTM into one vector
        query = Variable(torch.from_numpy(np.random.rand(1, query_size).astype(np.float32)), requires_grad=query_requires_grad)
        if device is not None:
            query = query.cuda(device=device)

        query_range = query_max - query_min
        assert query_range > 0
        query = query * query_range + query_min

        self.sda = ScaledDotAttention(d_k=hidden_size, d_v=hidden_size, d_q=query_size, model_dim=attention_dim)
        self.query = query

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(attention_dim * 4, linear_dim),
            torch.nn.Linear(linear_dim, linear_dim),
            torch.nn.Linear(linear_dim, n_out),
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
        ht, (_, _) = self.lstm(packed)
        ht_unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(ht)
        ht_unpacked = ht_unpacked.transpose(0, 1)
        
        sentence_vec = self.sda(K=ht_unpacked, V=ht_unpacked, Q=self.query)
        sentence_vec = sentence_vec.contiguous().view(sentence_vec.shape[0], -1)
        assert len(sentence_vec) == len(lens)

        # Reorder the sentences to the initial order
        sentence_vec = sentence_vec[inds_reverse]
        
        return sentence_vec

    def forward(self, vectors1, lens1, vectors2, lens2):
        sentence_vec1 = self._sentence_features(vectors1, lens1)
        sentence_vec2 = self._sentence_features(vectors2, lens2)
        
        features = torch.cat(
            [sentence_vec1, sentence_vec2, sentence_vec1 - sentence_vec2, sentence_vec1 * sentence_vec2],
            dim=1)
        pred = self.classifier(features)
        return F.log_softmax(pred, dim=1)


def test_lstm_att():
    bsz = 3
    seq1 = torch.from_numpy(np.random.rand(bsz, 20, 300,).astype(np.float32))
    seq2 = torch.from_numpy(np.random.rand(bsz, 20, 300,).astype(np.float32))
    model = LSTMAttention()

    outputs = model(seq1, [10, 20, 13], seq2, [12, 18, 20])
    assert outputs.shape == (bsz, 3)

test_lstm_att()