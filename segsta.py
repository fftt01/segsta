import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class imageEncode(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, act=None, image_size=256, mid_num=0,
                 res=None, kernel=3, group=1):
        super(imageEncode, self).__init__()
        if act is None:
            self.act = Swish()
        hidden_dim = out_channels
        self.same = in_channels == out_channels
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        if not self.same:
            self.besame = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = nn.Sequential(
            self.norm1,
            act,
            nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel, padding=kernel // 2, padding_mode='replicate',
                      groups=group),
            self.norm2,
            act,
            nn.Conv2d(hidden_dim, out_channels, kernel_size=kernel, padding=kernel // 2, padding_mode='replicate',
                      groups=group),
        )

    def forward(self, x):
        """
        batchsize, var, time, width, height
        """
        y = self.conv(x)
        if not self.same:
            x = self.besame(x)
        return x + y



class channelAttn(nn.Module):
    def __init__(self, in_channels=8, act=None, image_size=None, drop=0.1):
        super(channelAttn, self).__init__()
        if act is None:
            self.act = nn.ReLU()
        self.ffv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.ffq = nn.Sequential(
            # nn.Dropout(drop),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            # act,
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
        )
        self.ffk = nn.Sequential(
            # nn.Dropout(drop),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            # act,
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
        )
        self.norm = nn.GroupNorm(in_channels // 16, in_channels)
        self.in_channels = in_channels
        self.scale = (image_size[0]*image_size[1]) ** -0.5
        self.softmax = nn.Softmax(-1)
        self.image_size = image_size

    def forward(self, x):
        """
        batchsize, var*time, width, height
        """
        qkv = x
        qkv = self.norm(x)
        v = self.ffv(qkv)
        q = self.ffq(qkv)
        k = self.ffk(qkv)
        q = rearrange(q, "b n h w -> b n (h w)")
        k = rearrange(k, "b n h w -> b n (h w)")
        v = rearrange(v, "b n h w -> b n (h w)")
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        v = torch.matmul(attn, v)
        v = rearrange(v, 'b n (h w) -> b n h w', h=self.image_size[0], w=self.image_size[1])
        return v + x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class upl(nn.Module):
    def __init__(self, channels):
        super(upl, self).__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 3, 2, 0)
        self.norm = nn.GroupNorm(channels // 16, channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)[:, :, :-1, :-1]
        return x


class downl(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downl, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="replicate")
        return self.conv(x)


class STA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.k2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 2, padding_mode='replicate', groups=in_channels, dilation=2),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode='replicate', groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        )
        self.q2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 2, padding_mode='replicate', groups=in_channels, dilation=2),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode='replicate', groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        )
        self.v2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, padding_mode='replicate', )
        self.scale = nn.Parameter(torch.ones([1, in_channels, 1, 1]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        norm_x = self.norm(x)
        q = self.q2(norm_x)
        k = self.k2(norm_x)
        v = self.v2(norm_x)

        dots = q * k
        dots = self.sigmoid(dots)
        return v * dots * self.scale + shortcut



class segsta(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, image_size=None, act=Swish(), var_num=None, train_cur=None):
        super(segsta, self).__init__()
        self.out_channels = out_channels
        self.ifshortcut = in_channels == out_channels
        if isinstance(image_size, list):
            self.image_size = np.array(image_size)
        else:
            self.image_size = np.array([image_size, image_size])
        self.train_cur = train_cur
        self.var_num = var_num
        self.tnum = 5
        downc = [192, 256, 256, 384, 512]
        upc = [[downc[4], 384],
               [384, 384],
               [384, 256],
               [256, 256],
               [256, 256],
               ]
        self.timeembed = nn.Parameter(torch.randn(self.train_cur, 1, self.image_size[0], self.image_size[1]), requires_grad=True)
        in_channels += 1
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, downc[0], 3, 1, 1, padding_mode='replicate', ),
            # imageEncode(in_channels=downc[0], out_channels=downc[0], image_size=image_size, act=act),
        )
        self.down1 = nn.Sequential(
             downl(in_channels=downc[0], out_channels=downc[1]),
             imageEncode(in_channels=downc[1], out_channels=downc[1], act=act),
             # MoE(18, downc[1], act, topk, time),
             # imageEncode(in_channels=downc[1], out_channels=downc[1], image_size=image_size // 2, act=act),
             # spaceEncode(downc[1]),
             # nn.MaxPool2d(2),
        )
        self.down2 = nn.Sequential(
             downl(in_channels=downc[1], out_channels=downc[2]),
             imageEncode(in_channels=downc[2], out_channels=downc[2], act=act),
             # MoE(24, downc[2], act, topk, time),
             # imageEncode(in_channels=downc[2], out_channels=downc[2], image_size=image_size // 4, act=act),
             # spaceEncode(downc[2]),
             # nn.MaxPool2d(2),

        )
        self.down3 = nn.Sequential(
             downl(in_channels=downc[2], out_channels=downc[3]),
             imageEncode(in_channels=downc[3], out_channels=downc[3], act=act),
             # spaceEncode(downc[3]),
             # MoE(36, downc[3], act, topk, time),
             # imageEncode(in_channels=downc[3], out_channels=downc[3], image_size=image_size // 8, act=act),
             # nn.MaxPool2d(2),

        )
        n_experts = 6
        self.down4 = nn.Sequential(
             downl(in_channels=downc[3], out_channels=downc[4]),
             imageEncode(in_channels=downc[4], out_channels=downc[4], act=act),
             channelAttn(in_channels=downc[4], image_size=self.image_size//16),
             # MoE(48, downc[4], act, topk),
             imageEncode(in_channels=downc[4], out_channels=downc[4], act=act),
        )
        self.down4back = nn.Sequential(
            upl(upc[0][0]),
            imageEncode(in_channels=upc[0][0], out_channels=upc[0][1], act=act),
            STA(upc[0][1]),
            # MoE(40, upc[0][1], act, topk, time),
            imageEncode(in_channels=upc[0][1], out_channels=upc[0][1], act=act),
            # imageEncode(in_channels=upc[0][1], out_channels=upc[0][1], image_size=image_size // 8, act=act),
        )
        self.down3back = nn.Sequential(
            upl(upc[1][0]),
            imageEncode(in_channels=upc[1][0], out_channels=upc[1][1], act=act),
            STA(upc[1][1]),
            # MoE(36, upc[1][1], act, topk, time),
            imageEncode(in_channels=upc[1][1], out_channels=upc[1][1], act=act),
            # imageEncode(in_channels=upc[1][1], out_channels=upc[1][1], image_size=image_size // 4, act=act),
        )
        self.down2back = nn.Sequential(
            upl(upc[2][0]),
            imageEncode(in_channels=upc[2][0], out_channels=upc[2][1], act=act),
            STA(upc[2][1]),
            # MoE(24, upc[2][1], act, topk, time),
            imageEncode(in_channels=upc[2][1], out_channels=upc[2][1], act=act),
            # imageEncode(in_channels=upc[2][1], out_channels=upc[2][1], image_size=image_size // 2, act=act),
        )
        n_experts = 4
        self.down1back = nn.Sequential(
            upl(upc[3][0]),
            imageEncode(in_channels=upc[3][0], out_channels=upc[3][1], act=act),
            STA(upc[3][1]),
            imageEncode(in_channels=upc[3][1], out_channels=upc[3][1], act=act),
            # MoE(n_experts, upc[3][1], act, 1),
        )
        self.decode1 = nn.Sequential(
            imageEncode(in_channels=upc[4][0], out_channels=upc[4][1], act=act),
            STA(upc[4][1]),
            # MoE(24, upc[4][1], act, 1, time),
            imageEncode(in_channels=upc[4][1], out_channels=out_channels, act=act),
        )
        self.x3to = nn.Sequential(
            imageEncode(in_channels=downc[3], out_channels=upc[1][0], act=act),
            # spaceEncode(upc[1][0]),
            # MoE(16, downc[3], act, topk),
            # DropChannel(0.35),
        )
        self.x2to = nn.Sequential(
            # imageEncode(in_channels=downc[2], out_channels=downc[2], image_size=image_size // 4, act=act),
            imageEncode(in_channels=downc[2], out_channels=upc[2][0], act=act),
            # MoE(12, downc[2], act, topk),
            # spaceEncode(upc[2][0]),
            # spaceEncode(downc[2]),
            # DropChannel(0.4),
        )
        self.x1to = nn.Sequential(
            # imageEncode(in_channels=downc[1], out_channels=downc[1], image_size=image_size // 2, act=act),
            imageEncode(in_channels=downc[1], out_channels=upc[3][0], act=act),
            # spaceEncode(upc[3][0]),
            # MoE(9, downc[1], act, topk),
            # spaceEncode(downc[1]),
            # DropChannel(0.45),
        )
        self.xto = nn.Sequential(
            # imageEncode(in_channels=downc[0], out_channels=downc[0], image_size=image_size, act=act),
            imageEncode(in_channels=downc[0], out_channels=upc[4][0], act=act),
            # spaceEncode(upc[4][0]),
            # MoE(6, upc[4][0], act, topk),
            # nn.Identity()
            # spaceEncode(downc[0]),
            # DropChannel(0.5),
        )

    def forward(self, x, t):
        """
        batchsize, var, time, width, height
        """
        time_encode = self.timeembed[t]
        x = torch.cat([time_encode, x], dim=1)
        # z_q = self.memoget(c)
        x0 = self.encode(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x4 = self.down4back(x4)
        x3 = self.x3to(x3)
        x3 = self.down3back(x4+x3)
        x2 = self.x2to(x2)
        x2 = self.down2back(x3+x2)
        x1 = self.x1to(x1)
        x1 = self.down1back(x2+x1)
        x0 = self.xto(x0)
        result = self.decode1(x1+x0)
        return result

    def predict(self, x, start = 0, end = None):
        seg = self.out_channels // self.var_num
        if end is None:
            end = self.train_cur * seg
        remain = (end - start) % seg
        end_step = int((end - start - remain) / seg)
        start_step = int(start / seg)
        preds = []
        for i in range(start_step, end_step):
            t = torch.full((x.shape[0],), i, dtype=torch.int).to(x.device)
            pred = self.forward(x, t)
            pred = rearrange(pred, "b (t v) w h -> b t v w h", v=self.var_num)
            preds.append(pred)
        if remain > 0:
            t = torch.full((x.shape[0],), end_step+1, dtype=torch.int).to(x.device)
            pred = self.forward(x, t)
            pred = rearrange(pred, "b (t v) w h -> b t v w h", v=self.var_num)
            preds.append(pred)
        preds = torch.cat(preds, dim = 1)
        return preds

    def with_loss(self, x, y, t, std = None, mean = None, v = None, alpha=None):
        if len(x.shape) == 5:
            x = rearrange(x, "b t v w h -> b (t v) w h")
        pred = self.forward(x, t)
        if v is None:
            v = self.var_num
        pred = rearrange(pred, "b (t v) w h -> b t v w h", v = v)
        if std is not None:
            pred = pred * std + mean
        mse = (pred - y)**2
        loss = (((self.train_cur - t)**alpha) * mse.mean(dim=(1, 2, 3, 4)) / ((self.train_cur - t)**alpha).sum()).sum()
        return loss, pred