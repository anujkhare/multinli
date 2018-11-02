import torch
import torch.nn.functional as F


class CBOW(torch.nn.Module):
    def __init__(self, vector_dim=300, n_out=3) -> None:
        super().__init__()
        in_size = vector_dim * 4

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(in_size, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Tanh(),
        )
        self.classifier = torch.nn.Linear(1024, n_out)
        self.vector_dim = vector_dim

    def forward(self, data):
        vectors1, vectors2 = data
        #         print(vectors1.shape, vectors2.shape)
        s1, s2 = torch.sum(vectors1, dim=1), torch.sum(vectors2, dim=1)
        #         print(s1.shape, s2.shape)

        features = torch.cat([s1, s2, s1 - s2, s1 * s2], dim=1)
        features = self.feature_extractor(features)
        #         print(features.shape)
        pred = self.classifier(features)
        return F.log_softmax(pred, dim=1)