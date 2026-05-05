import torch
import torch.nn as nn
import torch.nn.functional as F
from .small_vit import ConvStem, EncoderBlock

class MIMDecoder(nn.Module):
    """Lightweight decoder for masked feature reconstruction."""
    def __init__(self, embed_dim, num_patches, num_layers=2, num_heads=4):
        super().__init__()
        self.num_patches = num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout=0.0) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, ids_keep):
        # x: visible encoder patch tokens (B, len_keep, C)
        # ids_keep: indices of these tokens in the original grid
        B, N_vis, C = x.shape
        total_patches = self.num_patches

        mask_tokens = self.mask_token.expand(B, total_patches, -1)
        full_x = mask_tokens.clone()
        full_x.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), x)

        full_x = full_x + self.decoder_pos_embed
        for blk in self.blocks:
            full_x = blk(full_x)
        full_x = self.norm(full_x)
        return self.pred(full_x)


class MIMSmallViT(nn.Module):
    def __init__(self, img_size=32, embed_dim=384, num_heads=4, num_blocks=4,
                 num_reg_tokens=4, conv_stages=3, mask_ratio=0.7, decoder_layers=2):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_reg_tokens = num_reg_tokens
        self.stem = ConvStem(img_size, embed_dim, num_stages=conv_stages)
        self.grid_size = self.stem.grid_size
        num_patches = self.grid_size ** 2
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + num_reg_tokens, embed_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, num_reg_tokens, embed_dim))
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout=0.0) for _ in range(num_blocks)
        ])
        self.decoder = MIMDecoder(embed_dim, num_patches, decoder_layers, num_heads)

    def forward(self, x):
        B = x.size(0)
        # 1. Stem -> feature grid
        features = self.stem(x)                                 # (B, C, H, W)
        H, W = self.grid_size, self.grid_size
        features_flat = features.flatten(2).transpose(1, 2)     # (B, N, C)
        N = self.num_patches
        target = features_flat.detach()                         # save full stem features as target
        mean = target.mean(dim=(0,1), keepdim=True)
        std = target.std(dim=(0,1), keepdim=True) + 1e-6
        target = (target - mean) / std

        # 2. Masking
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # 3. Select visible features
        x_vis = torch.gather(features_flat, dim=1,
                             index=ids_keep.unsqueeze(-1).expand(-1, -1, features_flat.size(-1)))

        # 4. Attach registers + positional embeddings
        reg = self.reg_tokens.expand(B, -1, -1)
        pos_reg = self.pos_embed[:, :self.num_reg_tokens, :]
        pos_patches = self.pos_embed[:, self.num_reg_tokens:, :]
        pos_vis = pos_patches.expand(B, -1, -1).gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, pos_patches.size(-1)))
        x_vis = torch.cat([reg, x_vis], dim=1)
        x_vis = x_vis + torch.cat([pos_reg.expand(B, -1, -1), pos_vis], dim=1)

        # 5. Encoder
        for blk in self.blocks:
            x_vis = blk(x_vis)

        # 6. Remove registers for decoder
        x_enc_patches = x_vis[:, self.num_reg_tokens:, :]   # (B, len_keep, C)

        # 7. Decoder predicts all stem features
        pred = self.decoder(x_enc_patches, ids_keep)        # (B, N, C)

        # 8. Loss on masked positions only
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)
        loss = F.mse_loss(pred[mask], target[mask])
        return loss