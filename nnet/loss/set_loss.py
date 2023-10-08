import torch.nn as nn


class SetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        """
        Compute the loss for set predictions
        :param outputs: output of the model
        :param targets: target of the model
        :return: loss
        """