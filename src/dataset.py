import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FILTER_MISSING_RATE = 0.80


def load_and_prepare(base_dir):
    meta_train_val = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_metadata_train_val(1).csv"))
    meta_test = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_metadata_test(1).csv"))
    meta = pd.concat([meta_train_val, meta_test], ignore_index=True)

    protein_train_val = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_proteome_raw_train_val.csv"))
    protein_test = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_proteome_raw_test.csv"))
    protein = pd.concat([protein_train_val, protein_test], ignore_index=True)

    meta = meta.set_index("sample_ID")
    protein = protein.set_index("sample_ID")

    common_ids = meta.index.intersection(protein.index)
    meta = meta.loc[common_ids]
    protein = protein.loc[common_ids]

    train_mask = meta["split_final"] == "train"
    train_protein = protein.loc[train_mask]
    missing_rate = train_protein.isna().mean()
    valid_protein_cols = missing_rate[missing_rate < FILTER_MISSING_RATE].index.tolist()
    protein = protein[valid_protein_cols]

    protein_log2 = np.log2(protein)

    cat_fields = ["Strains", "Medium", "Temperature",
                  "perturbation_no_concentration", "strain_role", "chemical_role"]
    encoders = {}
    for f in cat_fields:
        vals = ["<UNK>"] + sorted(meta[f].dropna().astype(str).unique().tolist())
        encoders[f] = {v: i for i, v in enumerate(vals)}

    return meta, protein_log2, encoders, valid_protein_cols


class PerturbationDataset(Dataset):
    def __init__(self, meta, protein_log2, split_name, encoders,
                 mask_prob=0.0, mask_value=0.0):
        """
        split_name: "train" | "val" | "test" | "val_strain_only" | ...
        """
        if split_name == "train":
            mask = meta["split_final"] == "train"
        elif split_name == "val":
            mask = meta["split_final"].str.startswith("val")
        elif split_name == "test":
            mask = meta["split_final"].str.startswith("test")
        else:
            mask = meta["split_final"] == split_name

        self.meta = meta.loc[mask].copy()
        self.protein_log2 = protein_log2.loc[self.meta.index]
        self.sample_ids = self.meta.index.tolist()
        self.encoders = encoders
        self.num_proteins = self.protein_log2.shape[1]

        self.mask_prob = mask_prob
        self.mask_value = mask_value

        self.cat_fields = ["Strains", "Medium", "Temperature",
                           "perturbation_no_concentration"]
        self.num_cats = {f: len(encoders[f]) for f in self.cat_fields}

    def __len__(self):
        return len(self.sample_ids)

    def encode_cat(self, field, value):
        v = str(value) if pd.notna(value) else "<UNK>"
        if v in self.encoders[field]:
            return self.encoders[field][v]
        return self.encoders[field]["<UNK>"]

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        row = self.meta.loc[sid]
        y_row = self.protein_log2.loc[sid].values.astype(np.float32)

        obs_mask = ~np.isnan(y_row)
        y_clean = y_row.copy()
        y_clean[~obs_mask] = 0.0

        input_vec = y_clean.copy()
        input_mask = obs_mask.copy().astype(np.float32)

        if self.mask_prob > 0.0:
            cand = np.where(obs_mask)[0]
            if len(cand) > 0:
                n_mask = int(np.ceil(len(cand) * self.mask_prob))
                if n_mask > 0:
                    mask_idx = np.random.choice(cand, size=n_mask, replace=False)
                    input_vec[mask_idx] = self.mask_value
                    input_mask[mask_idx] = 0.0

        cats = {f: self.encode_cat(f, row[f]) for f in self.cat_fields}

        return {
            "sample_id": sid,
            "x": torch.from_numpy(input_vec.astype(np.float32)),
            "x_mask": torch.from_numpy(input_mask),
            "y": torch.from_numpy(y_clean.astype(np.float32)),
            "y_mask": torch.from_numpy(obs_mask.astype(np.float32)),
            "strains": torch.tensor(cats["Strains"], dtype=torch.long),
            "medium": torch.tensor(cats["Medium"], dtype=torch.long),
            "temperature": torch.tensor(cats["Temperature"], dtype=torch.long),
            "chemical": torch.tensor(cats["perturbation_no_concentration"], dtype=torch.long),
        }
