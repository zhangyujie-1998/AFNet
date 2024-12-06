import torch
from torch.nn import functional as F
from utils.RankCompare.rankloss import SRCCLoss



class CTFLoss(torch.nn.Module):

    def __init__(self, **kwargs):
        super(CTFLoss, self).__init__()
        self.l2_w_c = 1
        self.l2_w_f = 1
        self.rank_w = 1
        self.hard_thred = 1
        self.srocc = SRCCLoss().cuda()

    
    def forward(self, output, gts):
        preds_c = output['score_coarse'].view(-1)
        preds_f = output['score_fine'].view(-1)
        gts = gts.view(-1)

        l2_c_loss = F.mse_loss(preds_c, gts) * self.l2_w_c
        l2_f_loss = F.mse_loss(preds_f, gts) * self.l2_w_f

        srocc_c = self.srocc(preds_c, gts)
        srocc_f = self.srocc(preds_f, gts)

        rank_diff = srocc_c  - srocc_f 
        loss_rank = torch.relu(rank_diff).mean() * self.rank_w

        loss_total = l2_c_loss + l2_f_loss + loss_rank + output['loss_dis']
    
        return loss_total
    
    
if __name__ == "__main__":
    
    mos = torch.randn(8,1).cuda()
    pre1 = torch.randn(8,1).cuda()
    pre2 = torch.randn(8,1).cuda()

    loss_function = CTFLoss().cuda()
    
    loss = loss_function(pre1, pre2, mos)
    print(loss)

