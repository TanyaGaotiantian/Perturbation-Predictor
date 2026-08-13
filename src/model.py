import torch
import torch.nn as nn
import torch.nn.functional as F


class PerturbationEncoder(nn.Module):
    """
    Encoder: protein expression vector + categorical context embeddings -> latent vector.

    Input dimensions:
        x:          (B, num_proteins)    # 4422
        x_mask:     (B, num_proteins)    # 1=observed, 0=missing/masked
        strains:    (B,)  Long
        medium:     (B,)  Long
        temperature:(B,)  Long
        chemical:   (B,)  Long
    """

    def __init__(self,
                 num_proteins=4422,
                 num_strains=7,
                 num_medium=3,
                 num_temp=3,
                 num_chemicals=58,
                 cat_emb_dim=32,
                 protein_hidden=2048,
                 latent_dim=512,
                 dropout=0.1):
        super().__init__()
        self.num_proteins = num_proteins
        self.latent_dim = latent_dim

        # Categorical embeddings (first slot = <UNK>)
        self.emb_strains = nn.Embedding(num_strains, cat_emb_dim)
        self.emb_medium = nn.Embedding(num_medium, cat_emb_dim)
        self.emb_temp = nn.Embedding(num_temp, cat_emb_dim)
        self.emb_chemical = nn.Embedding(num_chemicals, cat_emb_dim)

        cat_total = cat_emb_dim * 4  # 128

        # Protein branch: project (num_proteins + num_proteins [mask]) -> hidden
        self.protein_input_dim = num_proteins * 2  # concat x + mask
        self.protein_mlp = nn.Sequential(
            nn.Linear(self.protein_input_dim, protein_hidden),
            nn.LayerNorm(protein_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(protein_hidden, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Fusion: protein latent + cat_emb_total -> final latent
        fusion_in = latent_dim + cat_total
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, x_mask, strains, medium, temperature, chemical):
        """
        Returns:
            z: (B, latent_dim)  encoded latent vector
            cat_emb: (B, cat_total) for auxiliary use
        """
        e_s = self.emb_strains(strains)
        e_m = self.emb_medium(medium)
        e_t = self.emb_temp(temperature)
        e_c = self.emb_chemical(chemical)
        cat_emb = torch.cat([e_s, e_m, e_t, e_c], dim=-1)  # (B, 128)

        p_in = torch.cat([x, x_mask], dim=-1)  # (B, 4422*2)
        p_z = self.protein_mlp(p_in)            # (B, latent_dim)

        fused = torch.cat([p_z, cat_emb], dim=-1)  # (B, latent_dim + 128)
        z = self.fusion(fused)
        return z, cat_emb


class PerturbationMLP(nn.Module):
    """
    Decoder head: latent vector -> predicted protein expression.
    """

    def __init__(self, num_proteins=4422, latent_dim=512,
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.num_proteins = num_proteins
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_proteins),
        )

    def forward(self, z):
        return self.decoder(z)  # (B, num_proteins)


class PerturbationPredictor(nn.Module):
    """
    Full Encoder + MLP model.

    Input dims (for num_proteins=4422):
        x:             (B, 4422)   float32
        x_mask:        (B, 4422)   float32
        strains:       (B,)        int64   # ~7 classes
        medium:        (B,)        int64   # ~3 classes
        temperature:   (B,)        int64   # ~3 classes
        chemical:      (B,)        int64   # ~58 classes

    Encoder flow:
        protein_mlp input dim  = 4422 * 2 = 8844
        protein_mlp output dim = 512
        cat_emb dim            = 32 * 4  = 128
        fusion input dim       = 512 + 128 = 640
        fusion output (z) dim  = 512           <- MLP input dim

    MLP flow:
        input:  (B, 512)
        output: (B, 4422)

    Total params (approx): 30M
    """

    def __init__(self,
                 num_proteins=4422,
                 num_strains=7,
                 num_medium=3,
                 num_temp=3,
                 num_chemicals=58,
                 cat_emb_dim=32,
                 protein_hidden=2048,
                 latent_dim=512,
                 mlp_hidden=1024,
                 dropout=0.1):
        super().__init__()
        self.encoder = PerturbationEncoder(
            num_proteins=num_proteins,
            num_strains=num_strains,
            num_medium=num_medium,
            num_temp=num_temp,
            num_chemicals=num_chemicals,
            cat_emb_dim=cat_emb_dim,
            protein_hidden=protein_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = PerturbationMLP(
            num_proteins=num_proteins,
            latent_dim=latent_dim,
            hidden_dim=mlp_hidden,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, x_mask, strains, medium, temperature, chemical):
        z, cat_emb = self.encoder(x, x_mask, strains, medium, temperature, chemical)
        preds = self.decoder(z)
        return preds, z
