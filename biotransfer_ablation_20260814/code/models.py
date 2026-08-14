"""
models.py
统一模型模块，支持消融实验组件:
  - MLP:  基础 protein + context -> output
  - R:    Residual (protein input -> output skip connection)
  - L:    Latent State (context -> latent before decode)
  - C:    Cross-Attention (chemical × strain × protein)
  - P:    Protein Prior (protein-protein correlation bias)

所有变体共用同一 loss、同一 split、同一 lr、同一 epochs。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinEncoder(nn.Module):
    """protein expression + mask -> latent"""
    def __init__(self, num_proteins, hidden=2048, latent_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_proteins * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x, x_mask):
        return self.net(torch.cat([x, x_mask], dim=-1))


class ContextEncoder(nn.Module):
    """strain/medium/temp/chemical -> context latent

    strain: learned embedding (transferable)
    chemical: Morgan fingerprint (1024) or one-hot
    medium/temp: learned embedding
    """
    def __init__(self, num_strains, num_medium, num_temp,
                 chem_dim, strain_emb=32, medium_emb=16, temp_emb=16,
                 latent_dim=512, dropout=0.1):
        super().__init__()
        self.strain_emb = nn.Embedding(num_strains, strain_emb)
        self.medium_emb = nn.Embedding(num_medium, medium_emb)
        self.temp_emb = nn.Embedding(num_temp, temp_emb)

        context_in = strain_emb + medium_emb + temp_emb + chem_dim
        self.net = nn.Sequential(
            nn.Linear(context_in, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, strain_idx, medium_idx, temp_idx, chem_feat):
        s = self.strain_emb(strain_idx)
        m = self.medium_emb(medium_idx)
        t = self.temp_emb(temp_idx)
        cat = torch.cat([s, m, t, chem_feat], dim=-1)
        return self.net(cat)


class CrossAttention(nn.Module):
    """Cross-attention: protein latent (query) attends to context latent (key/value)"""
    def __init__(self, latent_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, protein_z, context_z):
        # (B, D) -> (B, 1, D)
        q = protein_z.unsqueeze(1)
        kv = context_z.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        out = self.norm(protein_z + attn_out.squeeze(1))
        return out


class ProteinPrior(nn.Module):
    """Protein prior: learnable bias vector + low-rank protein-protein interaction"""
    def __init__(self, num_proteins, rank=64):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_proteins))
        self.U = nn.Parameter(torch.randn(num_proteins, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, num_proteins) * 0.01)

    def forward(self, preds):
        low_rank = torch.sigmoid(preds @ self.U) @ self.V
        return preds + self.bias + 0.1 * low_rank


class DecoderMLP(nn.Module):
    """latent -> protein prediction"""
    def __init__(self, latent_dim, num_proteins, hidden=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_proteins),
        )

    def forward(self, z):
        return self.net(z)


class AblationModel(nn.Module):
    """
    Unified model supporting ablation flags.

    Components (additive):
      MLP  — base: protein_enc + context_enc -> concat -> decoder
      R    — Residual: add masked protein input to output
      L    — Latent State: fuse via latent space instead of raw concat
      C    — Cross-Attention: protein latent attends to context latent
      P    — Protein Prior: learnable bias + low-rank correction

    Config examples:
      MLP only:     use_residual=False, use_latent=False, use_cross=False, use_prior=False
      MLP + R:      use_residual=True, ...
      MLP + L + R:  use_residual=True, use_latent=True, ...
      C + L + R:    use_residual=True, use_latent=True, use_cross=True, ...
      C + L + R + P: all True
    """

    def __init__(self, num_proteins, num_strains, num_medium, num_temp,
                 chem_dim, strain_emb=32, medium_emb=16, temp_emb=16,
                 protein_hidden=2048, latent_dim=512, mlp_hidden=1024,
                 use_residual=False, use_latent=False,
                 use_cross=False, use_prior=False,
                 dropout=0.1):
        super().__init__()
        self.num_proteins = num_proteins
        self.use_residual = use_residual
        self.use_latent = use_latent
        self.use_cross = use_cross
        self.use_prior = use_prior

        self.protein_encoder = ProteinEncoder(num_proteins, protein_hidden, latent_dim, dropout)
        self.context_encoder = ContextEncoder(
            num_strains, num_medium, num_temp, chem_dim,
            strain_emb, medium_emb, temp_emb, latent_dim, dropout
        )

        if use_cross:
            self.cross_attn = CrossAttention(latent_dim, num_heads=8, dropout=dropout)

        if use_latent:
            # Fuse protein_z and context_z in latent space
            self.latent_fuse = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            decoder_in = latent_dim
        else:
            # Base MLP: concat protein_z + context_z
            decoder_in = latent_dim * 2

        self.decoder = DecoderMLP(decoder_in, num_proteins, mlp_hidden, dropout)

        if use_prior:
            self.protein_prior = ProteinPrior(num_proteins, rank=64)

        if use_residual:
            # Per-protein gated residual: gate starts at sigmoid(-5)≈0.007
            self.res_scale = nn.Parameter(torch.full((num_proteins,), -5.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, x_mask, strain_idx, medium_idx, temp_idx, chem_feat):
        p_z = self.protein_encoder(x, x_mask)
        c_z = self.context_encoder(strain_idx, medium_idx, temp_idx, chem_feat)

        if self.use_cross:
            p_z = self.cross_attn(p_z, c_z)

        if self.use_latent:
            z = self.latent_fuse(torch.cat([p_z, c_z], dim=-1))
        else:
            z = torch.cat([p_z, c_z], dim=-1)

        preds = self.decoder(z)

        if self.use_residual:
            # Per-protein gated residual: directly scale observed input
            gate = torch.sigmoid(self.res_scale)  # (num_proteins,)
            preds = preds + gate * x * x_mask

        if self.use_prior:
            preds = self.protein_prior(preds)

        return preds


# ============ Interaction Models (L5) ============

class InteractionHadamard(nn.Module):
    """strain + drug + strain⊙drug (element-wise product of embeddings)"""
    def __init__(self, num_proteins, num_strains, num_medium, num_temp,
                 chem_dim, strain_emb=32, medium_emb=16, temp_emb=16,
                 protein_hidden=2048, latent_dim=512, mlp_hidden=1024, dropout=0.1):
        super().__init__()
        self.num_proteins = num_proteins

        self.protein_encoder = ProteinEncoder(num_proteins, protein_hidden, latent_dim, dropout)

        self.strain_emb = nn.Embedding(num_strains, strain_emb)
        self.medium_emb = nn.Embedding(num_medium, medium_emb)
        self.temp_emb = nn.Embedding(num_temp, temp_emb)

        # Interaction: strain_emb ⊙ chem_proj
        self.chem_proj = nn.Linear(chem_dim, strain_emb)

        context_in = strain_emb + medium_emb + temp_emb + chem_dim + strain_emb  # extra strain_emb for interaction
        self.context_net = nn.Sequential(
            nn.Linear(context_in, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.decoder = DecoderMLP(latent_dim * 2, num_proteins, mlp_hidden, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, x_mask, strain_idx, medium_idx, temp_idx, chem_feat):
        p_z = self.protein_encoder(x, x_mask)

        s = self.strain_emb(strain_idx)
        m = self.medium_emb(medium_idx)
        t = self.temp_emb(temp_idx)
        c_proj = self.chem_proj(chem_feat)

        # Hadamard product: strain ⊙ drug
        interaction = s * c_proj

        cat = torch.cat([s, m, t, chem_feat, interaction], dim=-1)
        c_z = self.context_net(cat)

        z = torch.cat([p_z, c_z], dim=-1)
        return self.decoder(z)


class InteractionMLP(nn.Module):
    """strain + drug + Interaction MLP (learned interaction via separate MLP)"""
    def __init__(self, num_proteins, num_strains, num_medium, num_temp,
                 chem_dim, strain_emb=32, medium_emb=16, temp_emb=16,
                 protein_hidden=2048, latent_dim=512, mlp_hidden=1024,
                 interact_hidden=256, dropout=0.1):
        super().__init__()
        self.num_proteins = num_proteins

        self.protein_encoder = ProteinEncoder(num_proteins, protein_hidden, latent_dim, dropout)

        self.strain_emb = nn.Embedding(num_strains, strain_emb)
        self.medium_emb = nn.Embedding(num_medium, medium_emb)
        self.temp_emb = nn.Embedding(num_temp, temp_emb)

        # Interaction MLP: [strain_emb, chem_feat] -> interaction_emb
        interact_in = strain_emb + chem_dim
        self.interact_mlp = nn.Sequential(
            nn.Linear(interact_in, interact_hidden),
            nn.LayerNorm(interact_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interact_hidden, strain_emb),
            nn.LayerNorm(strain_emb),
            nn.GELU(),
        )

        context_in = strain_emb + medium_emb + temp_emb + chem_dim + strain_emb
        self.context_net = nn.Sequential(
            nn.Linear(context_in, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.decoder = DecoderMLP(latent_dim * 2, num_proteins, mlp_hidden, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, x_mask, strain_idx, medium_idx, temp_idx, chem_feat):
        p_z = self.protein_encoder(x, x_mask)

        s = self.strain_emb(strain_idx)
        m = self.medium_emb(medium_idx)
        t = self.temp_emb(temp_idx)

        # Interaction MLP
        interact_in = torch.cat([s, chem_feat], dim=-1)
        interaction = self.interact_mlp(interact_in)

        cat = torch.cat([s, m, t, chem_feat, interaction], dim=-1)
        c_z = self.context_net(cat)

        z = torch.cat([p_z, c_z], dim=-1)
        return self.decoder(z)
