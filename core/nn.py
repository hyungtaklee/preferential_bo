import math

import torch
from torch import nn

class NNRegressor(nn.Module):
    """

    """

    def __init__(self,
                 num_input_features: int):
        """

        """
        super().__init__()

        self.num_input_features = num_input_features

        # define a neural network
        self.layers = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self,
                f: torch.Tensor):
        """forward pass

        Args:
            f (torch.Tensor): input features

        Returns:
            torch.Tensor: scalar (1, ) correction value of the statistics.

        """
        return self.layers(f)

class CorrectionDataset(torch.utils.data.Dataset):
    """Generate/Create a dataset
    """
    def __init__(self, X, y, scale_data=True):
        """
        """
        # if not torch.is_tensor(X) and not torch.is_tensor(y):
        #     # Apply scaling if necessary
        #     if scale_data:
        #         X = StandardScaler().fit_transform(X)
        #     self.X = torch.from_numpy(X)
        #     self.y = torch.from_numpy(y)
        raise NotImplementedError("__init__() method hasn't been implemented")

    def __len__(self):
        """
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        """
        return self.X[i], self.y[i]
