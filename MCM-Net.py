import torch
import torch.nn as nn
import torch.nn.functional as F


class SPU(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()

        self.expand_conv = nn.Conv2d(
            in_channels,
            out_channels * (scale ** 2),
            kernel_size=1,
            bias=False
        )

        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        return x


class CBCC(nn.Module):
    def __init__(self, n_levels, chunk_dim, reduction=4, calib_scale=0.1):
        super().__init__()
        self.n_levels = n_levels
        self.chunk_dim = chunk_dim
        self.calib_scale = calib_scale

        in_dim = n_levels * chunk_dim
        mid_dim = max(in_dim // reduction, n_levels)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, in_dim, bias=True)
        )

        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, branch_list):

        B = branch_list[0].size(0)
        ctx = [
            branch.mean(dim=(-2, -1))
            for branch in branch_list
        ]
        ctx = torch.cat(ctx, dim=1)
        logits = self.fc(ctx)
        logits = logits.reshape(
            B,
            self.n_levels,
            self.chunk_dim,
            1,
            1
        )

        gates = 1.0 + self.calib_scale * torch.tanh(logits)

        calibrated = [
            branch_list[i] * gates[:, i]
            for i in range(self.n_levels)
        ]

        return calibrated


class DCSMCore(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        assert dim % n_levels == 0, f"dim={dim} must be divisible by n_levels={n_levels}"

        self.n_levels = n_levels
        self.chunk_dim = dim // n_levels

        self.branch_filters = nn.ModuleList([
            nn.Conv2d(
                self.chunk_dim,
                self.chunk_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.chunk_dim
            )
            for _ in range(self.n_levels)
        ])

        self.pool_logits = nn.Parameter(
            torch.zeros(self.n_levels - 1, 2, self.chunk_dim)
        )

        self.cross_branch_calib = CBCC(
            n_levels=self.n_levels,
            chunk_dim=self.chunk_dim,
            reduction=4,
            calib_scale=0.1
        )

        self.fusion = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)

        branch_outs = []

        for i in range(self.n_levels):
            if i > 0:
                p_size = (
                    max(h // (2 ** i), 1),
                    max(w // (2 ** i), 1)
                )

                s_max = F.adaptive_max_pool2d(xc[i], p_size)
                s_avg = F.adaptive_avg_pool2d(xc[i], p_size)

                weights = torch.softmax(self.pool_logits[i - 1], dim=0)                  

                w_max = weights[0].reshape(1, -1, 1, 1)
                w_avg = weights[1].reshape(1, -1, 1, 1)

                s = w_max * s_max + w_avg * s_avg
                s = self.branch_filters[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')

            else:
                s = self.branch_filters[i](xc[i])

            branch_outs.append(s)

        branch_outs = self.cross_branch_calib(branch_outs)

        out = torch.cat(branch_outs, dim=1)
        out = self.fusion(out)
        out = self.act(out) * x

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size=3,
            padding=1,
            groups=in_channel,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DCSM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.dcsm = DCSMCore(
            dim=out_channel,
            n_levels=4
        )
                                   
        self.conv2 = DepthwiseSeparableConv(
            in_channel=out_channel,
            out_channel=out_channel
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dcsm(x)
        x = self.conv2(x)
        return x

                                                                    
class SoftPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, eps=1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        patches = F.unfold(x, kernel_size=k, stride=s)                  
        patches = patches.reshape(B, C, k * k, -1)                          

        m = patches.max(dim=2, keepdim=True).values                 
        exp = torch.exp(patches - m)
        num = (patches * exp).sum(dim=2)                          
        den = exp.sum(dim=2) + self.eps                           
        out = num / den                                           

        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        out = out.reshape(B, C, out_h, out_w)
        return out


class SMKF(nn.Module):
                    
    def __init__(self, channels, use_k3=True, use_k5=True, use_k7=True, init_alpha=0.3):
        super().__init__()

        self.channel_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init_alpha)
        self.num_branches = int(use_k3) + int(use_k5) + int(use_k7)
        assert self.num_branches > 0, "At least one SMKF branch must be enabled."

        if use_k3:
            self.dw_3x3 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.dw_3x3 = None

        if use_k5:
            self.dw_5x5 = nn.Sequential(
                nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.dw_5x5 = None

        if use_k7:
            self.dw_7x7 = nn.Sequential(
                nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.dw_7x7 = None
                                     
        self.pw_conv = nn.Sequential(
            nn.Conv2d(channels * self.num_branches, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def channel_shuffle(x, groups: int):
        b, c, h, w = x.size()
        assert c % groups == 0
        cpg = c // groups
        x = x.reshape(b, groups, cpg, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.reshape(b, c, h, w)
        return x

    def forward(self, x):
        outs = []
        if self.dw_3x3 is not None:
            outs.append(self.dw_3x3(x))
        if self.dw_5x5 is not None:
            outs.append(self.dw_5x5(x))
        if self.dw_7x7 is not None:
            outs.append(self.dw_7x7(x))

        out = torch.cat(outs, dim=1)                             
        out = self.channel_shuffle(out, groups=self.num_branches)
        out = self.pw_conv(out)
        out = out * self.channel_scale

        return out


class SMKFBlock(nn.Module):
    def __init__(self, channels, use_k3=True, use_k5=True, use_k7=True, init_alpha=0.3):
        super().__init__()
        self.csmkdf = SMKF(channels, use_k3=use_k3, use_k5=use_k5, use_k7=use_k7, init_alpha=init_alpha)

    def forward(self, x):
        return self.csmkdf(x)


class CommonDifferentialMultiGrainedMixer(nn.Module):
    def __init__(self, channels, is_bn=True):
        super().__init__()

        self.fine_dw = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.medium_dw = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.proj_common = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.proj_diff = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.out_norm_act = nn.Sequential(
            nn.BatchNorm2d(channels) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        fine = self.fine_dw(x)
        medium = self.medium_dw(x)

        common = fine + medium
        diff = fine - medium

        common = self.proj_common(common)
        diff = self.proj_diff(diff)

        out = common + diff
        out = self.out_norm_act(out)

        return out


class CDMG(nn.Module):
    def __init__(self, channels=512, is_bn=True):
        super().__init__()

        self.local_embedding = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.multi_grained_mixer = CommonDifferentialMultiGrainedMixer(
            channels=channels,
            is_bn=is_bn
        )

    def forward(self, x):
        x = self.local_embedding(x)
        x = self.multi_grained_mixer(x)
        return x


class MCMEncoder(nn.Module):
    def __init__(self, in_channels=3, is_bn=True):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.pool2 = SoftPool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.pool3 = SoftPool2d(kernel_size=2, stride=2)
        self.block3_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.block3_csmkdf = SMKFBlock(256, use_k3=True, use_k5=True, use_k7=True, init_alpha=0.3)

        self.pool4 = SoftPool2d(kernel_size=2, stride=2)
        self.block4_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512) if is_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.block4_csmkdf = SMKFBlock(512, use_k3=True, use_k5=True, use_k7=True, init_alpha=0.5)

        self.pool5 = SoftPool2d(kernel_size=2, stride=2)

        self.block5 = CDMG(
            channels=512
        )

    def forward(self, x):
        f1 = self.block1(x)
        x2 = self.pool2(f1)
        f2 = self.block2(x2)

        x3 = self.pool3(f2)
        x3 = self.block3_conv(x3)
        f3 = self.block3_csmkdf(x3)

        x4 = self.pool4(f3)
        x4 = self.block4_conv(x4)
        f4 = self.block4_csmkdf(x4)

        x5 = self.pool5(f4)
        f5 = self.block5(x5)

        return f1, f2, f3, f4, f5


class MCMNet(nn.Module):
    def __init__(self, classes=3, in_channels=3):
        super().__init__()

        self.encoder = MCMEncoder(
            in_channels=in_channels,
            is_bn=True
        )

        self.up_4 = SPU(in_channels=512, out_channels=256, scale=2)
        self.conv_up3 = DCSM(256 + 512, 512)

        self.up_3 = SPU(in_channels=512, out_channels=256, scale=2)
        self.conv_up2 = DCSM(256 + 256, 256)

        self.up_2 = SPU(in_channels=256, out_channels=128, scale=2)
        self.conv_up1 = DCSM(128 + 128, 128)

        self.up_1 = SPU(in_channels=128, out_channels=64, scale=2)
        self.conv_up0 = DCSM(64 + 64, 64)

        self.seg_head = nn.Conv2d(64, classes, kernel_size=1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        assert h % 16 == 0 and w % 16 == 0, \
            f"Input size ({h}, {w}) must be divisible by 16."

        d0, d1, d2, d3, d4 = self.encoder(x)

        up_4 = self.up_4(d4)
        up_4 = torch.cat((up_4, d3), dim=1)
        up_3 = self.conv_up3(up_4)

        up_3 = self.up_3(up_3)
        up_3 = torch.cat((up_3, d2), dim=1)
        up_2 = self.conv_up2(up_3)

        up_2 = self.up_2(up_2)
        up_2 = torch.cat((up_2, d1), dim=1)
        up_1 = self.conv_up1(up_2)

        up_1 = self.up_1(up_1)
        up_1 = torch.cat((up_1, d0), dim=1)
        decoder_feat = self.conv_up0(up_1)

        cnn_logits = self.seg_head(decoder_feat)

        return cnn_logits
