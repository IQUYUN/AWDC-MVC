import torch
import torch.nn as nn

class TransformerViewFusion(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_heads, dropout, num_layers, num_views):
        super(TransformerViewFusion, self).__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(feature_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # optional output projection
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, zs):  
        # zs: list of [B, D], len = num_views
        B = zs[0].size(0)
        device = zs[0].device

        # [V, B, D] → [B, V, D]
        z = torch.stack(zs, dim=1)  # shape: [B, V, D]
        z = self.input_proj(z)      # → [B, V, embed_dim]

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(B, -1, -1)     # [B, 1, embed_dim]
        z = torch.cat([cls_tokens, z], dim=1)             # [B, V+1, embed_dim]

        # Transformer encoding
        z = self.encoder(z)  # [B, V+1, embed_dim]

        # output features
        H = z[:, 0]  # CLS token output as fused representation: [B, embed_dim]

        # attention weights from last layer, last head
        # Approximate view-level importance via cosine similarity to CLS
        cls_token_vec = H.unsqueeze(1)         # [B, 1, embed_dim]
        view_tokens = z[:, 1:]                 # [B, V, embed_dim]

        cos_sim = torch.cosine_similarity(view_tokens, cls_token_vec, dim=-1)  # [B, V]
        view_weights = torch.softmax(cos_sim, dim=1)                           # [B, V]

        # mean view weight across batch: [V]
        view_weights_mean = view_weights.mean(dim=0)

        return H, view_weights_mean
