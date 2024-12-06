import torch
import torch.nn as nn
import torch.nn.functional as F
from model.VisionTransformer import vit_base_patch16_224_in21k
from model.DRConv import DRConv2d


class AFQNet(nn.Module):
    def __init__(self):
        super(AFQNet, self).__init__()
        self.ViT = ViT()
        self.HyperNet = HyperNet()
    
    def forward(self,x):
        out = self.ViT(x)
        score_coarse = out['score_coarse']
        score_fine, loss_dis = self.HyperNet(out['img_cat'], out['f_guide'], out['f_global'])

        results = {}
        results['score_coarse'] = score_coarse
        results['score_fine'] = score_fine
        results['loss_dis'] = loss_dis
        return results


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.encoder = vit_base_patch16_224_in21k(pretrained=True)
        self.regc = nn.Sequential(nn.Linear(256,1))
        self.norm = nn.LayerNorm(196)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveAvgPool2d((2,2))
        self.conv = nn.Sequential(nn.Conv2d(12,64,kernel_size=1,stride=1),nn.ReLU())
        self.weight = (torch.ones((1, 1, 16, 16), dtype=torch.float32)/(16 * 16)).cuda()

    def forward(self,x):
        x_mask = x[:,:,4,:,:].unsqueeze(2).view(-1, 1, 224, 224)
        x = x[:,:,0:4,:,:]
     
        batch_size, num_images, channels, image_height, image_width = x.shape
        ratio_view = self.avgpool(x_mask).squeeze().view(-1,num_images,1).detach()
        ratio_view = ratio_view/torch.sum(ratio_view,dim=1).unsqueeze(1)

        ratio_patch = F.conv2d(x_mask, self.weight, stride=16)
        ratio_patch = ratio_patch.view(batch_size, num_images,1, 14, 14)
        ratio_patch = torch.cat([torch.cat([ratio_patch[:,i,:,:,:] for i in [0,1,2]], dim=3), \
                                    torch.cat([ratio_patch[:,i,:,:,:] for i in [3,4,5]], dim=3)], dim=2)

        x_cat = torch.cat([torch.cat([x[:,i,:,:,:] for i in [0,1,2]], dim=3), \
                            torch.cat([x[:,i,:,:,:] for i in [3,4,5]], dim=3)], dim=2)

        x = x.view(-1, channels, image_height, image_width)
        _, attn = self.encoder.forward_features(x)

        attn  = self.norm(attn)
        attn = attn.reshape(-1,12,14,14)
        attn_map = attn.reshape(batch_size, num_images, 12, 14, 14)
        attn_map = torch.cat([torch.cat([attn_map[:,i,:,:,:] for i in [0,1,2]], dim=3), \
                                torch.cat([attn_map[:,i,:,:,:] for i in [3,4,5]], dim=3)], dim=2)
        attn_map = attn_map * ratio_patch 

        f_g = self.conv(attn)
        f_g = self.maxpool(f_g)
        f_g = f_g.reshape(batch_size,num_images,256)
        f_g = torch.sum(f_g * ratio_view, dim=1)

        score_coarse = self.regc(f_g)

        out = {} 
        out['img_cat'] = x_cat
        out['score_coarse'] = score_coarse
        out['f_global'] = f_g
        out['f_guide'] = attn_map
 
        return out


class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()
        self.in_channels = 4
        self.out_channels =16
        self.drconv = DRConv2d(self.in_channels, self.out_channels, kernel_size=3, region_num=8, head_num=12)
        self.conv = nn.Conv2d(16,64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lda_pool = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),    
            nn.AdaptiveMaxPool2d((1,1))
        )
        self.regf = nn.Sequential(nn.Linear(512,1))
       
     
    def forward(self, x_cat, f_guide, f_g):
        f_l, _ = self.drconv(x_cat, f_guide)
        f_l = self.relu(f_l)
        f_l = self.relu(self.conv(f_l))
        f_l = self.maxpool(f_l)
        f_l = self.lda_pool(f_l).view(f_l.size(0),-1)

        loss_dis = torch.cosine_similarity(f_g, f_l)
        loss_dis = self.relu(loss_dis)
        loss_dis  = torch.mean(loss_dis,dim=0)

        f = torch.cat([f_g,f_l],1)
        score_fine = self.regf(f)
        
        return score_fine, loss_dis
    
