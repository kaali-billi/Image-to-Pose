import torch
import torch.nn as nn
from networks.layers import PyramidFeatureFusionV2, DctDownsample
from .newcrf_layers import NewCRF
from .swin_transformer import SwinTransformer
from .depth_update import DepthUpdateModule
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import math
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import timm


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, height * width, dim))

    def forward(self, x):
        # x: (B, N, D)
        return x + self.pe  # learned positional embedding added


class Max_Depth_Ada(nn.Module):
    """
    Original Logic : Single adabins predicts min and max
    New Loop : 1 adabins for max depth and 1 adabins for max-min = disp
                hard boundary for disp based on Dimensions of spacecraft (12m for DC)

    """

    def __init__(self, in_channels=512, embed_dim=1024, num_heads=8, height=7, width=7, dropout_p=0.15, min=0, max = 0):
        super().__init__()
        self.height = height
        self.width = width
        self.dropout_p = dropout_p

        self.q_proj = nn.Linear(in_channels, embed_dim)
        self.k_proj = nn.Linear(in_channels, embed_dim)
        self.v_proj = nn.Linear(in_channels, embed_dim)

        self.pos_enc = PositionalEncoding2D(embed_dim, height, width)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout_p)

        self.min_ada = AdaBins(feat_dim=embed_dim, n_bins=512, d_min=1.0, d_max=min)
        self.max_ada = AdaBins(feat_dim=embed_dim, n_bins=256, d_min=1.0, d_max=max)
        # bins = 256,dmax = 5 LRO
        # bins = 512, dmax = 12 DC dmin
        self.max_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
        )

        self.min_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
        )

        self.dropout = nn.Dropout(dropout_p)  # optional global dropout

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        N = H * W

        x1_flat = x1.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        x2_flat = x2.view(B, C, N).permute(0, 2, 1)

        Q = self.q_proj(x1_flat)
        K = self.k_proj(x2_flat)
        V = self.v_proj(x2_flat)

        Q = self.pos_enc(Q)
        K = self.pos_enc(K)

        attended, _ = self.attn(Q, K, V)  # (B, N, D)
        attended = self.dropout(attended)  # Apply dropout after attention
        pooled = attended.mean(dim=1)  # (B, D)

        min_logits = self.min_head(pooled)
        min_probs = F.softmax(min_logits, dim=1)
        min_centers = self.min_ada(pooled)  # (B, n_bins)
        min_depth = (min_probs * min_centers).sum(dim=1)  # (B,)

        max_logits = self.max_head(pooled)
        max_probs = F.softmax(max_logits, dim=1)
        max_centers = self.max_ada(pooled)
        max_depth = (max_probs * max_centers).sum(dim=1) + min_depth

        return min_depth, max_depth

class EfficientNetV2Encoder(nn.Module):
    def __init__(self):
        super(EfficientNetV2Encoder, self).__init__()
        base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Extract all layers except the classifier
        self.encoder = base_model.features  # Output: [B, 1280, 7, 7]

        # Optional: reduce channels to 512
        self.reduce_channels = nn.Conv2d(1280, 512, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((7,7))

    def forward(self, x):
        x = self.encoder(x)  # [B, 1280, 7, 7]
        x = self.pool(self.reduce_channels(x))  # [B, 512, 7, 7]
        return x


class AdaBins(nn.Module):
    def __init__(self, feat_dim=1024, n_bins=256, d_min=0.1, d_max=35):
        super().__init__()
        self.n_bins = n_bins
        self.d_min = d_min
        self.d_max = d_max

        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, n_bins),
            nn.Softmax(dim=1)  # attention-like weights
        )

    def forward(self, x):
        B = x.shape[0]
        bin_widths = self.predictor(x)  # (B, n_bins), softmaxed weights summing to 1
        bin_edges = bin_widths.cumsum(dim=1)  # cumulative sum
        bin_edges = torch.cat([torch.zeros(B, 1, device=x.device), bin_edges[:, :-1]], dim=1)  # prepend 0
        bin_centers = bin_edges + 0.5 * bin_widths  # midpoints between edges
        bin_centers = self.d_min + bin_centers * (self.d_max - self.d_min)  # scale to depth range
        return bin_centers  # (B, n_bins)


class DCDepth(nn.Module):
    """
    Depth network with dct. Replace the PPM head with PFF
    """

    def __init__(self, version=None, pretrained=None, ape: bool = False,
                 img_size: tuple = None, drop_path_rate: float = 0.2, drop_path_rate_crf: float = 0.,
                 seq_dropout_rate: float = 0., downsample_strategy: str = 'dct', **kwargs):
        super().__init__()

        self.patch_size = 8
        self.scale = 1
        self.img_size = img_size
        print(version[-2:], version[:-2])
        window_size = int(version[-2:])

        if version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f'Unknown version: {version}.')

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            pretrain_img_size=img_size,
            ape=ape
        )
        print(f'Backbone cfg: {backbone_cfg}.')

        embed_dim = 512  # Note, different embed_dim

        self.backbone = SwinTransformer(**backbone_cfg)

        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.hidden_dim = 192  # 128
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3],
                           num_heads=32, drop_path=drop_path_rate_crf)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2],
                           num_heads=16, drop_path=drop_path_rate_crf)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=self.hidden_dim, window_size=win, v_dim=v_dims[1],
                           num_heads=8, drop_path=drop_path_rate_crf)

        # build depth update module
        self.update = DepthUpdateModule(
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
            scale=self.scale,
            seq_drop_rate=seq_dropout_rate
        )

        self.decoder = PyramidFeatureFusionV2([8, 4, 2, 1], in_channels, embed_dim, downsample_strategy)
        self.project_hidden = nn.Conv2d(self.hidden_dim + in_channels[0], self.hidden_dim, 3, padding=1)
        self.project_context = DctDownsample(2, 5, 2, in_channels[0], in_channels[0])

        # Metric Range Prediction using Adaptive Bins
        self.rg_min = 35  # Maximum range of operations
        self.ob_max = 12  # Max dimension of object

        self.global_enc = EfficientNetV2Encoder()
        self.max_D = Max_Depth_Ada(min=self.rg_min, max=self.ob_max)


        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pretrained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        # self.decoder.init_weights()

    def parameters_1x(self):
        """
        Fine-tuning parameters
        :return:
        """
        yield from self.backbone.parameters()
        yield from self.global_enc.parameters()

    def parameters_5x(self):
        """
        Training parameters
        :return:
        """
        param_1x = set(self.parameters_1x())
        for param in self.parameters():
            if param not in param_1x:
                yield param

    def forward(self, imgs: torch.Tensor, rois: torch.Tensor, max_iters: int = None, return_freq_maps: bool = False):

        assert rois.shape[-2:] == self.img_size, f'Input image size {rois.shape} is not equal to {self.img_size}.'

        img_feats = self.global_enc(imgs)
        local_feats = self.backbone(rois)

        pff_out = self.decoder(*local_feats)
        e3 = self.crf3(local_feats[-1], pff_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(local_feats[-2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(local_feats[-3], e2)

        min_d, max_d = self.max_D(img_feats, pff_out)  # Metric Values of Min and Maximum depth in image
        context = self.project_context(local_feats[0])
        gru_hidden = torch.tanh(
            self.project_hidden(
                torch.cat([e1, context], 1)
            )
        )
        depths, freq = self.update(gru_hidden, max_iters=max_iters) # Normalized surface depth (0,1)
        if self.training:
            return depths, freq, min_d, max_d
        else:
            return depths[-1], min_d, max_d
