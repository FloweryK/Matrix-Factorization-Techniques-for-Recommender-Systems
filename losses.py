import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        mask = targets.ge(1)        # mask non-zero ratings only
        masked_inputs = torch.masked_select(inputs, mask)
        masked_targets = torch.masked_select(targets, mask)
        masked_mseloss = torch.mean((masked_targets - masked_inputs)**2)
        return masked_mseloss
