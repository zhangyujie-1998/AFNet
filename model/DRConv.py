import torch.nn.functional as F
import torch.nn as nn
import torch
import os

from torch.autograd import Variable, Function

class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) 
        return torch.sum(kernel * guide_mask, dim=1)
    
    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask 
        grad_guide = grad_output.clone().unsqueeze(1) * kernel 
        grad_guide = grad_guide.sum(dim=2) 
        softmax = F.softmax(guide_feature, 1) 
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True)) 
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    kernel_size = kernel.size()[3]
    pk = kernel.reshape(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, **kwargs, padding=int((kernel_size-1)/2), stride=1, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        kernel_size = kernel.size(3)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.reshape(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, padding=int((kernel_size-1)/2), stride=1, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out

class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, head_num = 12, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(head_num, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
        )
        
        self.conv_guide = nn.Conv2d(head_num, region_num, kernel_size=3, padding=1, **kwargs)
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply


    def forward(self, input, f_g):
        kernel = self.conv_kernel(f_g)  
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) 
        output = self.corr(input, kernel, **self.kwargs) 
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) 
        batch_size, region_num, _, width, height = output.shape
        guide_feature = self.conv_guide(f_g)
        guide_mask = guide_feature
        guide_feature = F.interpolate(guide_feature, size = (width,height), mode='bilinear', align_corners=False )
        output = self.asign_index(output, guide_feature)
        return output, guide_mask

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    B = 2
    in_channels = 4
    out_channels = 16
    size = 448
    conv = DRConv2d(in_channels, out_channels, kernel_size=3, region_num=4, head_num=768).cuda()
    conv.eval()
    input = torch.ones(B, 4, 448, 672).cuda()
    f_g = torch.ones(B,768,56, 84).cuda()
    
    output, guide_feature = conv(input, f_g)
    print(input.shape, output.shape)

    # flops, params
    from thop import profile
    from thop import clever_format
    
    class Conv2d(nn.Module):
        def __init__(self):
            super(Conv2d, self).__init__()
            self.conv = nn.Conv2d(4, out_channels, kernel_size=3)
        def forward(self, input):
            return self.conv(input)
    conv2 = Conv2d().cuda()
    conv2.train()
    macs2, params2 = profile(conv2, inputs=(input, ))
    macs, params = profile(conv, inputs=(input, f_g))
    print(macs2, params2)
    print(macs, params)