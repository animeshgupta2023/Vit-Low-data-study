import torch
import torch.nn as nn
import math

class ConvStem(nn.Module):
    """Small CNN that downsamples to a feature grid."""
    def __init__(self, img_size=32, embed_dim=384, num_stages=3):
        super().__init__()
        layers = []
        in_ch = 3
        chs = [embed_dim // 4, embed_dim // 2, embed_dim]  # for 3 stages
        for i, out_ch in enumerate(chs):
            if i < num_stages - 1:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            in_ch = out_ch
        self.stem = nn.Sequential(*layers)
        self.grid_size = img_size // (2 ** (num_stages - 1))

    def forward(self, x):
        return self.stem(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.size()
        qkv = self.qkv_proj(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        context = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.proj_drop(self.out_proj(context))


class FeedForward(nn.Module):  # SwiGLU
    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * expansion * 2 / 3)
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_value = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.sigmoid(self.w_gate(x)) * self.w_gate(x)  # Swish
        value = self.w_value(x)
        x = gate * value
        return self.dropout(self.w_out(x))


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = FeedForward(embed_dim, expansion, dropout)
        # LayerScale parameters
        self.ls1 = nn.Parameter(torch.full((embed_dim,), 1e-4))
        self.ls2 = nn.Parameter(torch.full((embed_dim,), 1e-4))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class PEG(nn.Module):
    """Positional Encoding Generator – depthwise conv on 2D feature grid."""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                stride=1, padding=kernel_size//2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_2d = self.dwconv(x_2d) + x_2d
        return x_2d.permute(0, 2, 3, 1).reshape(B, N, C)


class SmallViT(nn.Module):
    def __init__(self, img_size=32, num_classes=10, embed_dim=384, num_heads=4,
                 num_blocks=4, dropout=0.1, num_reg_tokens=4, conv_stages=3):
        super().__init__()
        self.num_reg_tokens = num_reg_tokens
        self.stem = ConvStem(img_size, embed_dim, num_stages=conv_stages)
        self.grid_size = self.stem.grid_size
        num_patches = self.grid_size ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + num_reg_tokens, embed_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, num_reg_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.peg = PEG(embed_dim)
        self.grid_H = self.grid_W = self.grid_size

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def get_features(self, x):
        B = x.size(0)
        x = self.stem(x)                      # (B, C, H, W)
        H, W = self.grid_H, self.grid_W
        x = x.flatten(2).transpose(1, 2)      # (B, N, C)

        reg = self.reg_tokens.expand(B, -1, -1)
        x = torch.cat([reg, x], dim=1)

        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == 0:
                patches = x[:, self.num_reg_tokens:]
                patches = self.peg(patches, H, W)
                x = torch.cat([x[:, :self.num_reg_tokens], patches], dim=1)

        x = x.mean(dim=1)                     # GAP (global average pooling)
        x = self.norm(x)
        return x

    def forward(self, x):
        features = self.get_features(x)
        return self.head(features), features