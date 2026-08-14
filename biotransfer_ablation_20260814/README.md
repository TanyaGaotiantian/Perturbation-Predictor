# BioTransfer-VCell：面向未知扰动泛化的酵母虚拟细胞知识迁移预测框架

## 一、项目概述

本项目参加世界人工智能开源大赛（GOAI）AI for Research 前沿探索赛道——虚拟细胞方向，任务为**基于酵母扰动蛋白质组数据的条件响应预测**。

核心科学问题：在有限扰动实验数据条件下，AI 模型能否学习具有生物意义的潜在细胞状态，并迁移到训练阶段未出现的新扰动组合？

我们的整体思路是：不直接学习 `Condition → Protein Response`，而是引入中间表示 `Condition → Latent Cellular State → Protein Response`，使模型学到扰动背后的细胞过程规律，从而获得对未知条件的泛化能力。

---

## 二、数据与任务

### 2.1 数据来源

| 数据文件 | 说明 |
|---------|------|
| `WAYB_WAYC_metadata_train_val(1).csv` | 训练/验证集元数据（样本ID、菌株、培养基、温度、扰动化学品、时间等） |
| `WAYB_WAYC_metadata_test(1).csv` | 测试集元数据 |
| `WAYB_WAYC_proteome_raw_train_val.csv` | 训练/验证集蛋白质组原始丰度矩阵 |
| `WAYB_WAYC_proteome_raw_test.csv` | 测试集蛋白质组原始丰度矩阵 |

### 2.2 数据规模

- **蛋白质数量**：4422 个（经缺失率 < 80% 过滤后保留）
- **菌株数量**：5 种（BAH、BAI、CEK、CGD、DHY210）
- **化学品数量**：46 种扰动剂（含 DMSO 对照）
- **培养条件**：2 种培养基（glucose / galactose）× 2 种温度（30°C / 37°C）
- **扰动时间**：15、30、60、90、120、240 分钟
- **数据来源**：WAYB（含3个批次：WAYB、WAYB_rep1、WAYB_rep2）+ WAYC

### 2.3 数据划分（OOD 评估设计）

赛事采用条件级 OOD 划分，确保测试集包含训练时未见的条件组合：

| Split | 含义 | 样本数 |
|-------|------|--------|
| `train` | 训练集 | 5920 |
| `val_seen` | 已见菌株 + 已见化学品 | ~589 |
| `val_strain_only` | 未见菌株，已见化学品 | 1547 |
| `val_chem_only` | 已见菌株，未见化学品 | 1065 |
| `val_both` | 未见菌株 + 未见化学品（完全 OOD） | 269 |
| `val_time` | 未见时间点 | 157 |

这种划分使我们能够分别评估模型在不同泛化难度下的表现，而非仅看整体指标。

---

## 三、实验流程（分步详解）

### L1/L2：基线建立

#### 做了什么

建立了两个基线模型作为后续改进的参照：

1. **Baseline 0 — 均值预测**：对每个蛋白质在训练集上取均值，作为所有条件的统一预测。这是最朴素的基线，用于确认任务难度。
2. **Baseline 1 — MLP**：将蛋白质输入（带掩码）与条件特征拼接后，通过多层感知机直接预测蛋白质组响应。

#### 如何实现

- 蛋白质输入：对原始丰度做 log2 变换，缺失值置零并附加二值掩码（mask）。
- 条件编码：菌株、培养基、温度、化学品均使用 one-hot 编码。
- 模型结构：`[protein + mask + condition_onehot] → MLP(2048→1024→4422)`。
- 损失函数：掩码均方误差（MSELoss），仅在非缺失位置计算损失。

#### 为什么这么做

均值基线确定了"不学习任何条件信息"时的性能下限；MLP 基线确定了"直接拟合条件-蛋白映射"的标准上限。所有后续改进都需要与这两个基线比较，才能确认提升是否真实。

#### 效果

MLP 基线在验证集上达到 RMSE=0.8218、R²=0.9123，显著优于均值预测，说明条件信息确实被模型利用。

---

### L3：Strain 可迁移表示

#### 做了什么

将菌株编码从 one-hot 改为**可学习嵌入（learned embedding，32 维）**，使菌株表示从离散无序的标识变为连续可迁移的向量。

#### 如何实现

```python
self.strain_emb = nn.Embedding(num_strains, strain_emb)  # strain_emb=32
```

- 菌株 ID 通过 `nn.Embedding` 映射为 32 维稠密向量。
- 培养基（16 维）和温度（16 维）同样使用 learned embedding。
- 嵌入向量与化学品特征拼接后，经过线性层映射到 512 维上下文隐向量。

#### 为什么这么做

One-hot 编码存在两个根本问题：

1. **维度固定**：one-hot 维度等于菌株数量，新增菌株需要扩展整个网络。
2. **无语义距离**：任意两个菌株之间的距离相同，无法体现生物学相似性。

Learned embedding 将离散类别映射到连续空间，训练过程中自动学习菌株间的语义关系。这在跨数据迁移场景（L6）中尤为关键——外部数据的菌株嵌入可以提供有意义的初始化，而 one-hot 无法做到。

#### 效果

Learned embedding 配合 Morgan 指纹的完整模型达到 RMSE=0.8218，与 one-hot 基线相当，但为后续迁移实验奠定了基础。

#### 优劣势

- **优势**：维度低（32 vs 5）、可迁移、可学习语义关系。
- **劣势**：需要足够训练数据才能学好嵌入；数据量过小时可能不如 one-hot。

---

### 化学品编码对比：One-hot vs Morgan 指纹

#### 做了什么

对比了两种化学品编码方式：

| 编码方式 | 维度 | 原理 |
|---------|------|------|
| One-hot | 57 | 每个化学品一个独热向量 |
| Morgan 指纹 | 1024 | 基于分子结构（SMILES → RDKit → Morgan FP, radius=2） |

#### 如何实现

- **One-hot**：`perturbation_no_concentration` 字段直接映射为 57 维 one-hot 向量。
- **Morgan 指纹**：通过 `dataset/chemical_encoder.py` 实现：
  1. 建立 57 个化学品名称到 SMILES 字符串的映射（`dataset/smiles_map.json`）。
  2. 使用 RDKit 将 SMILES 解析为分子对象。
  3. 计算半径为 2 的 Morgan 指纹（1024 位）。
  4. 对 9 个初始无效的 SMILES，通过 PubChem API 和 InChI 转换获取规范 SMILES，最终 56/57 成功编码。

```python
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
```

#### 为什么这么做

One-hot 编码将每种化学品视为完全独立的符号，丢失了分子的结构信息。例如，他莫昔芬和 4-羟基他莫昔芬在化学结构上高度相似，但 one-hot 编码后它们的距离与任意两个化学品相同。

Morgan 指纹基于分子子结构枚举，结构相似的化学品会共享更多指纹位。这使得模型能够利用化学结构先验，对**未见化学品**产生合理的泛化——结构相近的化学品可能引发相似的蛋白质组响应。

#### 效果

| 编码方式 | RMSE | R² | Pearson r |
|---------|------|-----|-----------|
| One-hot | 0.8516 | 0.9058 | 0.9518 |
| **Morgan FP** | **0.8419** | **0.9080** | **0.9530** |

Morgan 指纹在所有指标上均优于 one-hot，RMSE 降低 1.1%。这验证了分子结构信息的引入有助于提升预测性能，尤其在泛化到未见化学品时更为重要。

#### 改进过程

初始实现中 9 个化学品的 SMILES 无法被 RDKit 解析（如 Valinomycin 等大环化合物）。通过以下方式修复：
- 从 PubChem REST API 获取规范 SMILES。
- 对 InChI 格式的化合物使用 RDKit 的 InChI 转换功能。
- 最终 56/57 个化学品成功编码，仅 1 个返回全零向量（作为 fallback）。

---

### L4：残差分解

#### 做了什么

在基础 MLP 上引入**逐蛋白门控残差连接（Per-protein Gated Residual）**，将观测到的蛋白质输入直接接入输出端。

#### 如何实现

```python
# 门控初始化为 sigmoid(-5) ≈ 0.007，训练初期残差贡献极小
self.res_scale = nn.Parameter(torch.full((num_proteins,), -5.0))

# 前向传播中
gate = torch.sigmoid(self.res_scale)     # (num_proteins,)
preds = preds + gate * x * x_mask        # 仅对观测到的蛋白加残差
```

关键设计：
- **逐蛋白门控**：每个蛋白质有独立的残差权重，模型可以学习哪些蛋白需要强残差、哪些不需要。
- **小初始化**：门控初始值 ≈ 0.007，确保训练初期模型行为接近纯 MLP，避免残差项引入噪声。
- **掩码约束**：仅对观测到的蛋白质位置（`x_mask=1`）施加残差，缺失位置不受影响。

#### 为什么这么做

蛋白质组数据的一个显著特点是：同一细胞条件下，大部分蛋白的表达变化很小，只有少数关键蛋白发生显著响应。残差连接让模型学习**相对于观测基线的变化量（Δprotein）**，而非从零预测绝对值。

这类似于 ResNet 的思想：学习残差比学习完整映射更容易。但不同于标准 ResNet 的是，我们使用逐蛋白门控而非全局固定权重，因为不同蛋白质的动态范围和噪声水平差异很大。

#### 改进过程

初始实现使用直接相加（`preds = preds + x * x_mask`），导致 RMSE 飙升至 3.77，因为观测值的绝对量级（log2 后约 5-15）远大于模型预测的变化量。

改为**逐蛋白门控 + 小初始化**后，残差项在训练初期几乎不起作用，随训练逐步学习合适的权重，RMSE 降至 0.8093。

#### 效果

| 方法 | RMSE | R² | MAE | Pearson r |
|------|------|-----|-----|-----------|
| MLP | 0.8218 | 0.9123 | 0.6048 | 0.9553 |
| **MLP+R** | **0.8093** | **0.9150** | **0.5945** | **0.9566** |

残差分解在所有四个核心指标上均有提升，RMSE 降低 1.5%，且训练时间更短（131s vs 179s）。

#### 优劣势

- **优势**：参数量仅增加 4422（一个门控向量），计算开销可忽略；训练更稳定；对小变化蛋白特别有效。
- **劣势**：对于完全缺失的蛋白（无观测值），残差项不起作用，仍依赖 MLP 预测。

---

### L5：交互模块与渐进消融

#### 做了什么

设计了两组实验：

**实验一：交互模块对比**

| 方法 | 交互方式 |
|------|---------|
| Baseline (MLP) | 菌株嵌入与化学品特征简单拼接 |
| Hadamard | 菌株嵌入 ⊙ 化学品投影（逐元素乘积） |
| Interaction MLP | 专用交互 MLP 学习菌株-化学品联合表示 |

**实验二：渐进消融**

系统性评估各模块的贡献，按照以下顺序逐步叠加：

| 变体 | 组成 |
|------|------|
| MLP | 基础模型 |
| +R | + 残差分解 |
| L+R | + 潜在状态融合 |
| C+L+R | + 交叉注意力 |
| C+L+R+P | + 蛋白质先验 |

#### 如何实现

**Hadamard 交互**：
```python
c_proj = self.chem_proj(chem_feat)   # 化学品投影到菌株嵌入维度
interaction = s * c_proj             # 逐元素乘积
cat = torch.cat([s, m, t, chem_feat, interaction], dim=-1)
```

**Interaction MLP**：
```python
interact_in = torch.cat([s, chem_feat], dim=-1)
interaction = self.interact_mlp(interact_in)  # 256维 → 32维
```

**交叉注意力（C）**：蛋白质隐向量作为 Query，上下文隐向量作为 Key/Value：
```python
q = protein_z.unsqueeze(1)   # (B, 1, D)
kv = context_z.unsqueeze(1)  # (B, 1, D)
attn_out, _ = self.attn(q, kv, kv)
out = self.norm(protein_z + attn_out.squeeze(1))
```

**潜在状态融合（L）**：将蛋白质隐向量和上下文隐向量在潜在空间中融合，而非直接拼接：
```python
z = self.latent_fuse(torch.cat([p_z, c_z], dim=-1))  # 1024 → 512
```

**蛋白质先验（P）**：可学习偏置 + 低秩蛋白-蛋白互作校正：
```python
low_rank = torch.sigmoid(preds @ self.U) @ self.V  # U: (P, 64), V: (64, P)
return preds + self.bias + 0.1 * low_rank
```

#### 为什么这么做

**交互模块**的动机：MLP 直接拼接菌株和化学品特征后 flatten，丢失了二者之间的交互关系。生物学上，同一种药物在不同菌株中的效果可能截然不同（菌株-药物交互效应）。Hadamard 乘积是显式建模交互的最简单方式；Interaction MLP 则让模型自主学习更复杂的交互函数。

**渐进消融**的动机：我们提出了多个模块（R/L/C/P），需要确认每个模块是否真的有贡献。通过逐个叠加并对比，可以精确定位每个模块的边际收益，避免"堆模块但不涨点"的陷阱。

#### 效果

**交互模块对比**：

| 方法 | RMSE | R² | Pearson r |
|------|------|-----|-----------|
| Baseline (MLP) | 0.8218 | 0.9123 | 0.9553 |
| Hadamard | 0.8337 | 0.9098 | 0.9539 |
| Interaction MLP | 0.8248 | 0.9117 | 0.9548 |

**渐进消融**：

| 变体 | RMSE | R² | MAE | Pearson r |
|------|------|-----|-----|-----------|
| MLP | 0.8218 | 0.9123 | 0.6048 | 0.9553 |
| **+R** | **0.8093** | **0.9150** | **0.5945** | **0.9566** |
| L+R | 0.9136 | 0.8916 | 0.6847 | 0.9445 |
| C+L+R | 0.9331 | 0.8870 | 0.7069 | 0.9421 |
| C+L+R+P | 0.9338 | 0.8868 | 0.7076 | 0.9420 |

#### 分析

1. **交互模块未带来提升**：Hadamard 和 Interaction MLP 均不如 Baseline。这表明在 5 个 epoch 的训练预算下，额外的交互参数增加了优化难度，而简单的拼接已经足够捕获主要交互信息。

2. **残差（R）是最有效的模块**：+R 是所有变体中唯一优于基础 MLP 的，RMSE 降低 1.5%。

3. **复杂模块（L/C/P）在短训练下反而有害**：L+R、C+L+R、C+L+R+P 的 RMSE 均高于基础 MLP。原因分析：
   - 训练仅 5 个 epoch，复杂模块的参数尚未充分优化。
   - 潜在状态融合（L）引入了信息瓶颈（1024→512），可能丢失了直接拼接保留的信息。
   - 交叉注意力（C）在单 token 场景下退化为简单的加权求和，收益有限。
   - 蛋白质先验（P）的低秩矩阵需要更多训练数据才能学到有意义的蛋白-蛋白关系。

4. **最优策略**：在当前训练预算下，**MLP+R** 是最佳选择。复杂模块的潜力可能需要更长的训练和更大的数据量来释放。

#### 优劣势

- **优势**：消融实验提供了清晰的模块贡献排序；残差分解以最小代价获得最大收益。
- **劣势**：复杂模块（C/L/P）在短训练下未能发挥作用，需要更多训练预算验证其潜力；交互模块未带来预期提升。

---

### L6：跨数据迁移

#### 做了什么

验证外部数据预训练是否能提升目标数据集的泛化能力，特别是对未见条件的预测。

#### 实验设计

| 阶段 | 数据 | 说明 |
|------|------|------|
| 外部预训练 | WAYB + WAYB_rep1 + WAYB_rep2（train，2678 样本） | 模拟外部数据集 |
| 内部微调 | WAYC（train，3242 样本） | 官方训练集 |
| 评估 | WAYC（val splits，1214 样本） | 官方验证集 |

对比两种策略：
1. **Scratch**：从零开始，仅在 WAYC train 上训练 5 个 epoch。
2. **Pretrained**：先在 WAYB 上预训练 5 个 epoch，再在 WAYC train 上微调 5 个 epoch。

#### 如何实现

**数据集特定归一化**：
- WAYB 和 WAYC 分别在各自的 train 部分计算 per-protein 均值和标准差。
- 两个数据集独立做 z-score 归一化（mean=0, std=1），避免直接拼接不同量级的原始丰度。

```python
# WAYC: 独立 z-score
protein_norm[wayc_mask] = (protein_log2[wayc_mask] - mu_c) / sd_c
# WAYB: 独立 z-score
protein_norm[wayb_mask] = (protein_log2[wayb_mask] - mu_b) / sd_b
```

**预训练 → 微调**：
```python
# Stage 1: 在 WAYB 上预训练
model_pre = AblationModel(...)  # C+L+R+P
train_model(model_pre, ext_loader, val_loader, epochs=5)
pretrained_state = copy.deepcopy(model_pre.state_dict())

# Stage 2: 在 WAYC train 上微调（继续训练同一模型）
train_model(model_pre, int_loader, val_loader, epochs=5)
```

#### 为什么这么做

**为什么用 WAYB 模拟外部数据**：真实的外部酵母扰动蛋白质组数据集需要复杂的 protein ID 映射（UniProt/SGD）和跨平台归一化。作为方法验证，我们利用同一研究中不同批次的数据（WAYB vs WAYC）来模拟跨数据集场景——它们共享相同的 protein ID（无需映射），但来自不同实验批次，存在自然的分布偏移。

**为什么做数据集特定归一化**：不同实验批次的蛋白质丰度存在系统性偏移（仪器灵敏度、上样量差异等）。如果直接拼接原始丰度，模型会学习到批次效应而非生物学信号。独立 z-score 归一化消除了批次间的量级差异，同时保留了组内相对变化模式。

**为什么用 C+L+R+P 而非 MLP+R**：L6 的目标是验证"复杂模型 + 外部预训练"是否能超越"简单模型从零训练"。如果复杂模块在外部数据上学到有意义的潜在状态表示，微调后可能在目标数据上表现更好。

#### 改进过程

初始实现中，WAYB 的归一化做了 z-score 后又平移到 WAYC 的均值（`+ mu_c / sd_c`），导致 WAYB 目标值在 5-15 范围，而 WAYC 目标值在 0 附近。预训练损失飙升至 1932，模型完全无法收敛。

修复后两个数据集独立 z-score，均归一化到 mean=0, std=1，预训练损失正常降至 0.29。

#### 效果

**整体对比**：

| 方法 | RMSE | R² | MAE | Pearson r |
|------|------|-----|-----|-----------|
| Scratch | 0.6728 | 0.4770 | 0.5015 | 0.6930 |
| **Pretrained** | **0.6611** | **0.4951** | **0.4902** | **0.7069** |
| Δ | +1.74% | +1.81% | +2.25% | +2.00% |

**分 split 对比**（ΔRMSE = scratch − pretrained，正值表示预训练更好）：

| Split | Scratch RMSE | Pretrained RMSE | ΔRMSE | 提升幅度 |
|-------|-------------|-----------------|-------|---------|
| val_both (OOD) | 0.7043 | 0.6977 | +0.0066 | +0.94% |
| val_strain_only | 0.6968 | 0.6877 | +0.0091 | +1.31% |
| **val_chem_only** | 0.5865 | 0.5638 | **+0.0227** | **+3.87%** |
| **val_time** | 0.6139 | 0.5948 | **+0.0190** | **+3.10%** |

#### 分析

1. **预训练在所有 split 上均优于从零训练**，验证了 BioTransfer 的核心假设：外部数据预训练能提升目标数据集的泛化能力。

2. **最大提升出现在 val_chem_only（+3.87%）和 val_time（+3.10%）**。这具有重要意义：
   - val_chem_only 是未见化学品的泛化测试，预训练使模型在 WAYB 上学到了更通用的化学品-蛋白响应模式。
   - val_time 是未见时间点的泛化测试，预训练帮助模型学到了更稳健的时间动态表示。

3. **val_both（完全 OOD）的提升较小（+0.94%）**，因为同时外推菌株和化学品是最困难的场景，预训练的边际收益有限。

4. **注意**：L6 的 RMSE（~0.66）与 L4/L5（~0.81）不直接可比，因为 L6 使用了 z-score 归一化（目标值 std≈1），而 L4/L5 使用 log2 变换（目标值范围更大）。两者的关键对比是各自内部的 scratch vs pretrained。

#### 优劣势

- **优势**：验证了跨数据迁移的有效性；未见化学品提升最大，符合 BioTransfer 的科学故事；流程可复用于真实外部数据。
- **劣势**：WAYB 和 WAYC 来自同一研究，分布偏移较小；真实外部数据需要 protein ID 映射和更复杂的归一化；提升幅度在 1-4% 之间，绝对值不大。

---

### L7：深度条件级评估

#### 做了什么

对最佳模型架构进行多维度的细粒度评估：

1. **分 split 评估**：在 val_seen / val_strain_only / val_chem_only / val_both / val_time 上分别评估 4 个模型变体（R / L+R / C+L+R / C+L+R+P）。
2. **条件级分析**：按菌株、化学品、时间、温度、培养基分别计算 RMSE，找最强/最弱条件。
3. **蛋白质级解释**：计算每个蛋白质的预测 RMSE 和 Pearson 相关性，找最准/最差预测的蛋白。
4. **模块贡献分析**：在 val_both（完全 OOD）上对比去掉各模块后的性能变化。

#### 如何实现

为每个模型变体独立训练（相同超参数），然后在验证集上收集每个样本的预测值和真值。按元数据字段分组计算指标：

```python
# 条件级分析：按菌株分组
groups = meta.loc[sample_ids, "Strains"]
for gval in groups.unique():
    idxs = [i for s, i in sid_to_idx.items() if groups[s] == gval]
    metrics = compute_metrics(y[idxs], preds[idxs], ym[idxs])

# 蛋白质级分析：逐蛋白计算
for j in range(num_proteins):
    mask_j = ym[:, j]
    rmse = np.sqrt(np.mean((y[mask_j, j] - preds[mask_j, j]) ** 2))
```

#### 效果

**分 split 评估（RMSE）**：

| Split | R | L+R | C+L+R | C+L+R+P |
|-------|---|-----|-------|---------|
| val_seen | 0.7945 | 0.8518 | 0.8784 | 0.8794 |
| val_strain_only | 0.7815 | 0.8419 | 0.8768 | 0.8773 |
| val_chem_only | 0.8811 | 0.9572 | 1.0042 | 1.0045 |
| **val_both (OOD)** | **0.8808** | 0.9437 | 0.9884 | 0.9884 |
| val_time | 0.7945 | 0.8518 | 0.8784 | 0.8794 |

**条件级发现**：

| 条件维度 | 最易预测 | 最难预测 | RMSE 差异 |
|---------|---------|---------|----------|
| 菌株 | BAI (0.894) | BAH (1.023) | 0.129 |
| 时间 | 30 min (0.918) | 240 min (0.961) | 0.043 |
| 温度 | 30°C (0.929) | 37°C (0.938) | 0.009 |
| 培养基 | glucose (0.925) | galactose (0.942) | 0.017 |

**蛋白质级发现**：
- 4422 个蛋白质，中位数 RMSE = 0.807，均值 = 0.889
- **最易预测**：TRP2 (RMSE=0.28)、RPT2 (0.28)、SPF1 (0.29) — 这些蛋白表达变化幅度小、噪声低
- **最难预测**：RPL31B (RMSE=3.76)、ERR1 (3.74)、RPS26A (3.50) — 多为核糖体蛋白（RPL/RPS），表达动态范围大、响应剧烈

**模块贡献（val_both OOD）**：

| 模型 | RMSE | vs C+L+R+P |
|------|------|-----------|
| **R only** | **0.8808** | −0.108 (更好) |
| L+R | 0.9437 | −0.045 (更好) |
| C+L+R | 0.9884 | 0.000 |
| C+L+R+P | 0.9884 | — |

#### 分析

1. **R only 在所有 split 上都是最优模型**。这进一步证实了 L4/L5 的结论：在当前训练预算下，简单模型 + 残差是最有效的组合。

2. **val_chem_only 是所有模型最困难的 split**（RMSE ~0.88-1.00），因为未见化学品需要模型从分子结构泛化，这是最本质的 OOD 挑战。

3. **菌株间预测难度差异显著**：BAI 最易（可能因为训练样本最多，1816 个），BAH 最难。这提示模型对样本量少的菌株泛化能力不足。

4. **时间动态的规律性**：30 min 最易预测，240 min 最难。短时间扰动的蛋白质组变化更规律，长时间扰动可能触发复杂的级联响应，增加了预测难度。

5. **核糖体蛋白是最难预测的蛋白群**。这具有生物学意义——核糖体蛋白在应激响应中变化剧烈（可达 10 倍以上），且受多重调控通路影响，模型难以精确捕捉其动态范围。

6. **模块贡献的悖论**：在 OOD 场景下，去掉复杂模块反而更好（R < L+R < C+L+R = C+L+R+P）。这说明复杂模块在有限训练下过拟合了训练分布的特异性，损害了泛化能力。蛋白质先验（P）在当前实现下几乎无贡献（C+L+R ≈ C+L+R+P），可能需要更多训练数据或更好的先验设计。

---

### L8：最终收敛

#### 做了什么

汇总所有实验结果，生成最终报告和可视化。

#### 生成的文件

| 文件 | 说明 |
|------|------|
| `final_report.txt` | 7 张结果表（主结果表、消融表、OOD 表、迁移表、条件分析表、模块贡献表、蛋白质级表） |
| `final_summary.json` | 结构化结果数据 |
| `plot_L8_architecture.png` | 模型架构图 |

#### 最终结论

1. **最佳单模型**：MLP+R（残差分解），RMSE=0.8093，R²=0.9150。
2. **最佳迁移策略**：WAYB 预训练 → WAYC 微调，在所有 OOD split 上均优于从零训练，未见化学品提升 3.87%。
3. **关键发现**：残差分解是最有效的模块；复杂模块（交叉注意力、潜在融合、蛋白先验）在 5 epoch 训练预算下未能发挥作用，甚至损害 OOD 泛化；外部预训练的有效性验证了 BioTransfer 框架的科学故事。

---

## 四、统一实验控制

为确保对比的公平性，所有实验严格固定以下变量：

| 变量 | 值 | 说明 |
|------|-----|------|
| 数据划分 | train / val（含子 split） | 所有实验使用同一 split |
| 损失函数 | MSELoss (masked) | 仅在非缺失位置计算 |
| 学习率 | 3e-4 | AdamW 优化器 |
| Epochs | 5 | L6 预训练另加 5 epoch |
| Batch size | 64 | — |
| Weight decay | 1e-4 | — |
| Mask probability | 0.15 | 随机掩码 15% 观测蛋白 |
| 随机种子 | 42 | 确保可复现 |
| 化学品编码 | Morgan FP (1024 bit, radius=2) | 所有后续实验统一使用 |
| 菌株编码 | Learned embedding (32d) | 所有后续实验统一使用 |
| 潜在维度 | 512 | — |

---

## 五、代码结构

```
base1/
├── dataset/                          # 数据与编码模块
│   ├── chemical_encoder.py           # Morgan 指纹编码器（RDKit）
│   ├── smiles_map.json               # 57 个化学品 → SMILES 映射
│   └── dataset.py                    # 基础数据集类
│
├── models.py                         # 统一模型定义（AblationModel + 交互模型）
│   ├── ProteinEncoder                # 蛋白编码器：protein + mask → latent
│   ├── ContextEncoder                # 上下文编码器：strain/medium/temp/chem → latent
│   ├── CrossAttention                # 交叉注意力：protein Q × context KV
│   ├── ProteinPrior                  # 蛋白先验：bias + low-rank 校正
│   ├── DecoderMLP                    # 解码器：latent → protein prediction
│   ├── AblationModel                 # 统一消融模型（R/L/C/P 可选开关）
│   ├── InteractionHadamard           # Hadamard 交互模型
│   └── InteractionMLP                # Interaction MLP 模型
│
├── compare_chemical_encoding.py      # L3: one-hot vs Morgan 对比实验
├── run_ablation.py                   # L4+L5: 统一消融实验
├── run_transfer.py                   # L6: 跨数据迁移实验
├── run_conditional_eval.py           # L7: 条件级深度评估
├── generate_final_report.py          # L8: 最终报告生成
│
├── plot_ablation.py                  # L4/L5/L7 绘图
├── plot_transfer.py                  # L6 绘图
├── plot_conditional_eval.py          # L7 绘图
│
├── ablation_results.json             # L4/L5 消融结果
├── transfer_results.json             # L6 迁移结果
├── conditional_eval_results.json     # L7 条件评估结果
├── chemical_encoding_comparison.json # 化学品编码对比结果
├── final_report.txt                  # L8 最终报告
├── final_summary.json                # L8 结构化结果
│
├── plot_L4_residual_comparison.png   # L4: MLP vs MLP+R 对比图
├── plot_L5_interaction_comparison.png# L5: 交互模块对比图
├── plot_L5_ablation_progressive.png  # L5: 渐进消融图
├── plot_L6_overall_comparison.png    # L6: 迁移整体对比图
├── plot_L6_per_split_comparison.png  # L6: 分 split 迁移对比图
├── plot_L6_improvement_delta.png     # L6: 迁移提升 ΔRMSE 图
├── plot_L7_per_split_models.png      # L7: 分 split × 模型对比图
├── plot_L7_conditional_rmse.png      # L7: 条件级 RMSE 图
├── plot_L7_module_contribution.png   # L7: 模块贡献图
├── plot_L8_architecture.png          # L8: 架构图
└── ...
```

---

## 六、复现指南

### 环境依赖

```
Python >= 3.8
PyTorch >= 2.0
rdkit
scikit-learn
scipy
pandas
numpy
matplotlib
```

### 运行顺序

```bash
# 1. 化学品编码对比（L3）
python compare_chemical_encoding.py

# 2. 统一消融实验（L4+L5）
python run_ablation.py

# 3. 绘制消融对比图（L4+L5+L7 per-split）
python plot_ablation.py

# 4. 跨数据迁移实验（L6）
python run_transfer.py

# 5. 绘制迁移对比图（L6）
python plot_transfer.py

# 6. 条件级深度评估（L7）
python run_conditional_eval.py

# 7. 绘制条件评估图（L7）
python plot_conditional_eval.py

# 8. 生成最终报告（L8）
python generate_final_report.py
```

### 注意事项

- 所有实验默认在 CPU 上运行（可通过修改 `device = torch.device("cuda")` 切换到 GPU）。
- 完整流程（L3-L8）在 CPU 上约需 30-40 分钟。
- 随机种子固定为 42，结果可复现。
- 数据文件需放在项目根目录下。

---

## 七、结果总结

### 主结果表

| 方法 | RMSE(↓) | R²(↑) | MAE(↓) | Pearson(↑) | 参数量 |
|------|---------|-------|--------|-----------|--------|
| MLP (baseline) | 0.8218 | 0.9123 | 0.6048 | 0.9553 | 26.4M |
| **MLP+R (最优)** | **0.8093** | **0.9150** | **0.5945** | **0.9566** | 26.4M |
| Hadamard | 0.8337 | 0.9098 | 0.6113 | 0.9539 | 26.4M |
| Interaction MLP | 0.8248 | 0.9117 | 0.6049 | 0.9548 | 26.7M |

### 迁移学习表

| 方法 | RMSE(↓) | R²(↑) | ΔRMSE vs scratch |
|------|---------|-------|-----------------|
| Scratch | 0.6728 | 0.4770 | — |
| **Pretrained** | **0.6611** | **0.4951** | **+1.74%** |

### OOD 泛化表（MLP+R 模型）

| Split | RMSE | R² | 泛化难度 |
|-------|------|-----|---------|
| val_seen | 0.7945 | 0.9182 | 低 |
| val_strain_only | 0.7815 | 0.9201 | 中 |
| val_chem_only | 0.8811 | 0.9000 | 高 |
| val_both (OOD) | 0.8808 | 0.9002 | 最高 |

---

## 八、优劣势分析

### 优势

1. **统一的实验控制**：所有实验固定 split/loss/lr/epoch/seed，确保对比公平，结论可信。
2. **残差分解简单有效**：以最小参数代价（+4422 参数）获得最大收益（RMSE −1.5%），且训练更稳定。
3. **迁移学习故事成立**：外部预训练在所有 OOD split 上均有提升，特别是未见化学品提升 3.87%，验证了 BioTransfer 框架的核心假设。
4. **细粒度评估全面**：从 split 级、条件级、蛋白质级三个层次分析模型表现，提供了丰富的可解释性洞察。
5. **分子结构先验有效**：Morgan 指纹优于 one-hot，验证了引入化学结构信息的价值。

### 劣势与改进方向

1. **复杂模块未发挥预期作用**：交叉注意力（C）、潜在融合（L）、蛋白先验（P）在 5 epoch 训练下反而降低性能。改进方向：
   - 增加训练 epoch 至 20-50，让复杂模块充分优化。
   - 使用学习率预热（warmup）和分层学习率，避免复杂模块早期不稳定。
   - 重新设计蛋白先验，引入真实的 GO/KEGG 通路知识而非低秩近似。

2. **交互模块未带来提升**：Hadamard 和 Interaction MLP 均不如 baseline。可能原因：
   - 菌株-化学品交互在当前数据规模下不是主要瓶颈。
   - 交互模块引入的额外参数增加了优化难度。
   - 改进方向：尝试冻结 base 模型后仅训练交互模块，或使用更轻量的交互设计。

3. **迁移实验使用同源数据**：WAYB 和 WAYC 来自同一研究，分布偏移有限。真实外部数据集（如 Y3K、Stefaniak et al.）的迁移效果有待验证，需要解决 protein ID 映射和跨平台归一化问题。

4. **训练预算有限**：5 个 epoch 对于 26M 参数的模型偏少，复杂模块的潜力可能未被充分释放。

5. **缺少 pathway 级分析**：当前仅做了 protein 级分析，未将蛋白映射到 GO/KEGG 通路进行系统性的通路级评估。这是后续解释模型预测的重要方向。

---

## 九、关键结论

1. **残差分解是核心模块**：在所有实验中，MLP+R 始终是最优的单模型配置。逐蛋白门控残差以极低代价学习 Δprotein，优于直接预测绝对值。

2. **外部预训练有效但收益集中在特定 OOD 场景**：迁移学习对未见化学品（+3.87%）和未见时间（+3.10%）提升最大，对完全 OOD（val_both）提升有限（+0.94%）。这提示外部数据的主要价值在于补充化学品和条件的覆盖，而非菌株覆盖。

3. **模型复杂度与训练预算需匹配**：在有限训练预算下，简单模型（MLP+R）优于复杂模型（C+L+R+P）。复杂模块的价值需要更多数据和更长训练来验证。

4. **核糖体蛋白是最难预测的蛋白群**：RPL/RPS 家族蛋白在应激响应中变化剧烈，是模型的主要误差来源。针对高动态范围蛋白的特殊处理（如分箱预测或 log-ratio 预测）可能是改进方向。

5. **化学品结构先验对 OOD 泛化有帮助**：Morgan 指纹优于 one-hot，且迁移学习对未见化学品提升最大，说明分子结构信息是泛化到新化学品的关键。
