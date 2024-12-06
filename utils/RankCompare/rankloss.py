import torch
import torch.nn as nn
from utils.RankCompare.fast_soft_sort.pytorch_ops import soft_rank

class PLCCLoss(nn.Module):

    def __init__(self):
        super(PLCCLoss, self).__init__()

    def forward(self, input, target):
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)
        self.loss = torch.sum(input0 * target0) / (torch.sqrt(torch.sum(input0 ** 2))
                                                   * torch.sqrt(torch.sum(target0 ** 2)))
        return self.loss

class SRCCLoss(nn.Module):
    def __init__(self):
        super(SRCCLoss, self).__init__()

    def forward(self, input, target):

        if input.numel() == 1 and target.numel() == 1:
            return torch.tensor(1.0, device=input.device, requires_grad=True)
        input = input.flatten().unsqueeze(0)
        target = target.flatten().unsqueeze(0)
        input = soft_rank(input, regularization_strength=0.1).squeeze()
        target = soft_rank(target, regularization_strength=0.1).squeeze()
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)
        self.loss = torch.sum(input0 * target0) / (torch.sqrt(torch.sum(input0 ** 2))
                                                   * torch.sqrt(torch.sum(target0 ** 2)))
        return self.loss
    