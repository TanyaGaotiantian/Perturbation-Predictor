"""
chemical_encoder.py
将 perturbation_no_concentration (化学品名) 通过 SMILES -> Morgan fingerprint 编码。

用法:
    from dataset.chemical_encoder import ChemicalEncoder

    encoder = ChemicalEncoder(smiles_map_path="dataset/smiles_map.json", n_bits=1024)
    fp = encoder.encode("DMSO")           # -> np.ndarray (1024,)
    fp = encoder.encode("Unknown Drug")    # -> np.zeros(1024)
"""
import json
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class ChemicalEncoder:
    """SMILES -> Morgan fingerprint 编码器。"""

    def __init__(self, smiles_map_path=None, n_bits=1024, radius=2):
        self.n_bits = n_bits
        self.radius = radius
        self.smiles_map = {}
        self._fp_cache = {}

        if smiles_map_path and os.path.exists(smiles_map_path):
            with open(smiles_map_path, "r", encoding="utf-8") as f:
                self.smiles_map = json.load(f)

        self._precompute_all()

    def _precompute_all(self):
        """预先计算所有化学品的 Morgan fingerprint 并缓存。"""
        for name, smiles in self.smiles_map.items():
            if smiles:
                fp = self._smiles_to_morgan(smiles)
                if fp is not None:
                    self._fp_cache[name] = fp

    def _smiles_to_morgan(self, smiles):
        """将单个 SMILES 字符串转为 Morgan fingerprint numpy 数组。"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.n_bits
        )
        return np.asarray(fp, dtype=np.float32)

    def encode(self, chemical_name):
        """
        根据化学品名返回 Morgan fingerprint。

        Args:
            chemical_name: perturbation_no_concentration 字段值

        Returns:
            np.ndarray (n_bits,), float32。未找到时返回全零向量。
        """
        if chemical_name in self._fp_cache:
            return self._fp_cache[chemical_name]
        return np.zeros(self.n_bits, dtype=np.float32)

    @property
    def output_dim(self):
        return self.n_bits

    def summary(self):
        total = len(self.smiles_map)
        valid = len(self._fp_cache)
        return f"ChemicalEncoder: {valid}/{total} chemicals encoded, dim={self.n_bits}"
