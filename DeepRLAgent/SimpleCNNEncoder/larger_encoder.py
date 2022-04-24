import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, num_classes, state_size):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        # Code Change: Architecture Changes
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(state_size, 256, 1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_classes, 1),
        )
        #self.encoder = nn.Conv1d(state_size, num_classes, 1)
        # self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.encoder(x.unsqueeze(2))
        x = x.squeeze() if len(x.squeeze().shape) > 1 else x.squeeze().unsqueeze(0)
        # return self.out(x)
        return x
