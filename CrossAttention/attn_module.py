
import torch
from einops import rearrange
import torch.nn as nn
import math
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from CASCADE.decoders import *
from CASCADE.pvtv2 import pvt_v2_b2

class Img_cross_attn(nn.Module):
    ## input has shape [B, C, H, W]
    ## output has shape [B, C, H, W]
    
    def __init__(self, num_heads=8, query_embed_dim=512, kv_embed_dim=None):
        super(Img_cross_attn, self).__init__()
        if kv_embed_dim is None:
            kv_embed_dim = query_embed_dim
        
        self.attn = nn.MultiheadAttention(embed_dim=query_embed_dim, num_heads=num_heads, kdim=kv_embed_dim, vdim=kv_embed_dim, batch_first=True)
        # self.q  = nn.LazyLinear(query_embed_dim)
        self.pos_q = nn.Parameter(torch.randn(1,query_embed_dim, 100, 100))   
        self.pos_k = nn.Parameter(torch.randn(1, kv_embed_dim, 100, 100))   
        # self.pos_v = nn.Parameter(torch.randn(1, num_patches, embed_dim))   
    def _get_pos_embed(self, base_pos, h, w):
        pos_embed = torch.nn.functional.interpolate(base_pos, size=(h,w), mode='bilinear', align_corners=False)
        return rearrange(pos_embed, 'b c h w -> b (h w) c')
    
    def forward(self, q, k, v):
        h_q, w_q = q.shape[-2], q.shape[-1]
        h_k, w_k = k.shape[-2], k.shape[-1]

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')
        positional_q = q + self._get_pos_embed(self.pos_q, h=h_q, w=w_q)
        positional_k = k + self._get_pos_embed(self.pos_k, h=h_k, w=w_k)

        output, attn_weights = self.attn(query=positional_q, key=positional_k, value=v)
        output = rearrange(output, 'b (h w) c -> b c h w', h = h_q, w = w_q)
        return output, attn_weights


class Cross_CASCADE_Decoder(nn.Module):
    def __init__(self, channels=[512,320,128,64], num_heads=8):
        super(Cross_CASCADE_Decoder,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=32)
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SpatialAttention()

        self.attn_modules = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.attn_modules.append(Img_cross_attn(num_heads=8, query_embed_dim=channels[0], kv_embed_dim=channels[i + 1]))

    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        for i,item in enumerate(self.attn_modules):
            d4, _ = item(d4, skips[i], skips[i])
        
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1

class PVT_W_Cross_CASCADE(nn.Module):
    def __init__(self, pvt_backbone_path='./pretrained_pth/pvt/pvt_v2_b2.pth', n_class=1):
        super(PVT_W_Cross_CASCADE, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = pvt_backbone_path
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = Cross_CASCADE_Decoder(channels=[512, 320, 128, 64])
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4


if __name__ == "__main__":
    attn = Img_cross_attn(num_heads=8, query_embed_dim=128, kv_embed_dim=256)

    q = torch.randn(10, 128, 88, 88)
    k = torch.randn(10, 256, 11, 11)
    v = torch.randn(10, 256, 11, 11)
    output, attn_weights = attn(q, k, v)
    print(output.shape) ## (10, 128, 88 * 88)
    print(attn_weights.shape) ## 88 * 88, 11 * 11
    
    flops = FlopCountAnalysis(attn, inputs=(q, k, v))
    print(f"Total FLOPs: {flops.total() / 1e9} GFLOPs")
    # 2. Compute Parameters (neatly formatted table)
    print(parameter_count_table(attn))
