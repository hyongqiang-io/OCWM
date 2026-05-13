# Experimental Setup: Multi-View LPWM + JEPA + GNN-Mamba3

## 1. Research Questions

| RQ | Question | Resolved by |
|----|----------|-------------|
| RQ1 | Does JEPA latent prediction outperform pixel reconstruction for particle dynamics learning? | Ablation: LPWM vs LPWM-JEPA |
| RQ2 | Does GNN-Mamba3 outperform spatio-temporal Transformer for particle dynamics? | Ablation: LPWM-GNN-Tx vs LPWM-GNN-Mamba3 |
| RQ3 | Does explicit GNN interaction context improve over pure Mamba3 temporal modeling? | Ablation: LPWM-SSM vs LPWM-GNN-Mamba3 |
| RQ4 | Does multi-view 3D grounding improve over single-view representation? | Ablation: LPWM-JEPA vs LPWM-3D-JEPA |
| RQ5 | Does particle-based decomposition outperform slot-based for fine-grained spatial dynamics? | External: vs C-JEPA, HCLSM |
| RQ6 | Does object-centric decomposition outperform flat patch representations? | External: vs V-JEPA 2-AC |
| RQ7 | Is GNN interaction context functionally distinct from HCLSM's causal graph discovery? | Qualitative + quantitative analysis |

---

## 2. Datasets

### Primary Benchmarks

| Dataset | Type | Views | Resolution | Horizon | Purpose |
|---------|------|-------|------------|---------|---------|
| **Sketchy** | Robot manipulation, rigid objects | 1 (+ synthetic 2nd) | 128×128 | T=20 | Main benchmark, matches LPWM paper |
| **CLEVRER** | Physics simulation, collisions | 1 (+ synthetic 2nd) | 128×128 | T=20 | Counterfactual reasoning (C-JEPA comparison) |
| **PushT** (OXE) | Robot push manipulation | 1–2 | 64–128×128 | T=10–20 | HCLSM comparison (same benchmark) |
| **Balls Occlusion** | Synthetic, occlusion-heavy | 1 | 64×64 | T=10 | Fast ablation dataset |

### Multi-View Data Protocol

For datasets without natural second views, synthesize view2 via:
1. Known camera transform T_{1→2} (rotation + baseline translation)
2. Rendered from same physics engine (CLEVRER, Balls)
3. Real second camera when available (PushT OXE subset, Sketchy real-robot variant)

All baselines receive **identical multi-view input**. Single-view baselines are explicitly labeled as such in the paper.

### Data Splits

- Train / Val / Test: 80% / 10% / 10% by episode
- No frame-level leakage: splits are episode-level
- Multi-step evaluation uses held-out episodes only (no frames seen during training)

---

## 3. Model: LPWM-GNN-Mamba3 (Proposed)

### Architecture

```
Encoder:   DLP3D (shared, EMA copy for JEPA target)
           K=30 particles: (z_pos ∈ R^2, z_depth ∈ R^1, z_scale ∈ R^2,
                             z_features ∈ R^{16}, z_obj_on ∈ R^1)

Dynamics:  N=3 interleaved GNN-Mamba3 blocks
           - GNN: 2-layer MPNN, edge feat = concat(z_i, z_j, |p_i - p_j|_3D)
                  attentional aggregation → context c^{(i)} ∈ R^{d_c}
           - Mamba3: input = concat(z^{(i)}, c^{(i)}) ∈ R^{d+d_c}
                     d_state = 64, d_conv = 4, expand = 2

Context:   DLPContext (unchanged from LPWM baseline)
           4-layer Transformer, 8 heads, dim=512

Decoder:   DLPDecoder (used only during lambda_rec > 0 phase)
```

### Key Dimensions

| Hyperparameter | Value | Source |
|---------------|-------|--------|
| Particles K | 30 | Sketchy config (n_kp_enc=30) |
| Prior KP | 64 | All configs |
| Timestep horizon T | 20 | Sketchy config |
| pint_dim (Mamba hidden) | 512 | Sketchy config (pint_dim=512) |
| GNN edge dim d_e | 128 | New; ~0.25× pint_dim |
| Mamba d_state | 64 | New; sweep 32/64/128 |
| Image size | 128×128 | Sketchy config |
| Feature dim | 16 | Increased from 4 (Sketchy) for richer 3D encoding |

### Training Configuration

```
Optimizer:      Adam, lr=8e-5, betas=(0.9, 0.999), eps=1e-6
Scheduler:      CosineAnnealingLR after warmup_epoch=1
Batch size:     6 (Sketchy), 8 (Balls/CLEVRER) — same as LPWM baseline
Epochs:         150 (same as LPWM baseline for fair comparison)
Warmup frames:  num_static_frames=1

Loss weights:
  beta_kl:      0.08
  beta_rec:     1.0 (during annealing phase)
  beta_dyn:     0.2
  lambda_cv:    1.0 (cross-view loss)
  lambda_jepa:  1.0

lambda_rec annealing:
  Epoch 0–30:   lambda_rec = 1.0  (joint JEPA + reconstruction)
  Epoch 30–80:  lambda_rec linearly 1.0 → 0.1
  Epoch 80+:    lambda_rec = 0.0  (pure JEPA)

EMA decay:      tau = 0.996  (V-JEPA default)
```

---

## 4. Baselines

### 4a. Component Ablations (Internal, trained on same data)

All ablations use K=30 particles, T=20, same optimizer and epochs.

| ID | Name | Dynamics | Training Loss | 3D | Purpose |
|----|------|----------|--------------|-----|---------|
| A0 | **LPWM** | SpatioTemporalTransformer (6L-8H) | Pixel recon + KL | No | Upper bound of prior work |
| A1 | **LPWM-3D** | SpatioTemporalTransformer (6L-8H) | Pixel recon + KL + L_cv | Yes | Isolate: multi-view 3D grounding value |
| A2 | **LPWM-JEPA** | SpatioTemporalTransformer (6L-8H) | JEPA + KL | No | Isolate: JEPA training paradigm value |
| A3 | **LPWM-3D-JEPA** | SpatioTemporalTransformer (6L-8H) | JEPA + KL + L_cv | Yes | JEPA + 3D, no Mamba/GNN |
| A4 | **LPWM-SSM** | Mamba3 only (no GNN) | JEPA + KL + L_cv | Yes | Isolate: Mamba3 vs Transformer |
| A5 | **LPWM-GNN-Tx** | GNN + Transformer | JEPA + KL + L_cv | Yes | Isolate: GNN value independent of Mamba3 |
| **A6** | **LPWM-GNN-Mamba3** | GNN + Mamba3 | JEPA + KL + L_cv | Yes | **Proposed model** |

Ablation chain for each RQ:
- RQ1: A0 → A2 (JEPA paradigm, no 3D)
- RQ2: A3 → A5 → A6 (GNN-Tx vs GNN-Mamba3, controlling for 3D+JEPA)
- RQ3: A4 → A6 (GNN contribution to Mamba3)
- RQ4: A2 → A3 (3D grounding on top of JEPA)

### 4b. External Paradigm Comparisons

| Model | Paper | Architecture | Signal | Training | Code |
|-------|-------|-------------|--------|----------|------|
| **C-JEPA** | arXiv 2602.11389 (LeCun group, 2026-02) | Object-centric + JEPA + object-level masking | Causal masking | JEPA | galilai-group/cjepa |
| **HCLSM** | arXiv 2603.29090 (2026-03) | Slot Attention + 3-level SSM/Tx + GNN causal graph | Pixel recon (2-stage) | Supervised | rightnow-ai/hclsm |
| **V-JEPA 2-AC** | Meta AI 2025 | Flat patch ViT + JEPA + action condition | Masked patch prediction | JEPA | meta-research/jepa |
| **PlaySlot** | LPWM original paper comparison | Slot-based + causal Transformer | Pixel recon | Supervised | (from LPWM paper) |

**Distinction vs HCLSM (critical for reviewers):**
HCLSM's GNN learns a static causal interaction graph (which object types affect which).
Our GNN produces dynamic per-frame interaction context vectors fed into Mamba3.
HCLSM's GNN output: adjacency structure A ∈ {0,1}^{K×K}.
Our GNN output: context c_t^{(i)} ∈ R^{d_c}, updated every frame.
Empirical test: evaluate on scenes with time-varying interaction topology (e.g., particles forming and breaking clusters) where static graphs are insufficient.

**Distinction vs C-JEPA:**
C-JEPA self-supervised signal: same-frame inter-object masking (spatial).
Our signal: cross-frame cross-view geometric projection (spatiotemporal + 3D).
Expected advantage: 3D grounding enables better generalization to novel viewpoints; C-JEPA does not explicitly model 3D geometry.

### 4c. Planning Evaluation (Downstream)

HWM (arXiv 2604.03208) is used as a **planning layer**, not an architecture baseline.

Evaluation protocol:
- Attach HWM hierarchical planner on top of: (1) LPWM-GNN-Mamba3, (2) original LPWM, (3) V-JEPA 2-AC
- Compare downstream task success rate to isolate world model quality from planner quality
- Metrics: pick-&-place success rate (%), planning compute (FLOPs per decision)

VLA comparison (data-efficiency track only):
- Fine-tune π0 (or OpenVLA) on N% of training episodes
- Train our WM + MPC on same N% of episodes
- Compare: few-shot sample efficiency curve (N = {1%, 5%, 10%, 50%, 100%})
- Do NOT compare raw success rates without data-matching — this is the fairness trap from the design doc

---

## 5. Evaluation Metrics

### World Model Quality

| Metric | Definition | Horizon |
|--------|-----------|---------|
| **MSE** | Per-pixel mean squared error on rollout frames | T=5, 10, 20 |
| **FVD** | Fréchet Video Distance on 16-frame rollouts | T=16 |
| **LPIPS** | Perceptual similarity (VGG) | T=5, 10, 20 |
| **SSIM** | Structural similarity | T=5, 10, 20 |

Report mean and std over 3 seeds. Report separately for short (T≤5) and long (T>10) horizons — Mamba3's advantage is expected to surface primarily at long horizon.

### Object-Centric Quality

| Metric | Definition | Notes |
|--------|-----------|-------|
| **ARI** | Adjusted Rand Index for object segmentation | Requires GT object masks |
| **Tracking accuracy** | IoU between predicted and GT particle trajectories | Threshold IoU=0.2 (from LPWM config: iou_thresh=0.2) |
| **Particle specialization** | obj_on variance across K particles (higher = better specialization) | Proxy for object discovery quality |
| **Depth consistency** | L1 error on unprojected 3D positions vs GT depth (where available) | 3D-specific |

### Cross-View Reconstruction (3D-Specific)

| Metric | Definition |
|--------|-----------|
| **Cross-view PSNR/SSIM** | Quality of view2 reconstruction from view1 encoded particles |
| **Depth rank consistency** | Spearman correlation of predicted depth order vs GT order |
| **Geometric equivariance error** | ||T_{1→2}(z_3D^{v1}) - z_3D^{v2}||_2 normalized by scene scale |

### Efficiency

| Metric | How to measure |
|--------|---------------|
| **Rollout FLOPs** | Count MACs for T=20 autoregressive rollout |
| **Rollout latency** | Wall-clock time for T=20 rollout, batch=1, single GPU |
| **Training throughput** | Samples/sec during training (same hardware for all models) |
| **Memory footprint** | Peak GPU memory during training, batch=6 |

Target: GNN-Mamba3 should match or exceed Transformer prediction quality at ≤ 50% rollout FLOPs.

### Downstream Control

| Metric | Task | Protocol |
|--------|------|---------|
| **Success rate (%)** | Sketchy push / PushT | 100 test episodes, report mean ± std |
| **Planning FLOPs** | All tasks | FLOPs per action decision (MPC + world model forward) |
| **Sample efficiency** | All tasks | Success rate vs number of training episodes |

---

## 6. Fairness Controls

### Capacity Matching

All ablation variants (A0–A6) must have matched total parameter counts within ±10%.

| Variant | Dynamics params target | Strategy |
|---------|----------------------|---------|
| LPWM (Transformer) | ~15M (6L × 512d × 8H) | Baseline |
| LPWM-SSM (Mamba3 only) | ~15M | Increase d_state or layers to match |
| LPWM-GNN-Tx | ~15M | GNN ~3M + Transformer ~12M |
| LPWM-GNN-Mamba3 | ~15M | GNN ~3M + Mamba3 ~12M |

Report exact parameter counts for all variants in the paper.

### Training Budget Matching

- Fixed: **150 epochs** and **same dataset size** for all variants
- Do NOT match wall-clock time (Mamba3 is faster per step, which is the efficiency claim being tested)
- Do NOT match FLOPs/epoch (this conflates training efficiency with model quality)
- Rationale: "same data traversal" is the standard for architecture comparison

### Multi-View Input Matching

- All variants receive the **same two-view input** during training
- Single-view ablations are explicitly labeled and excluded from 3D-metric comparisons
- HCLSM and C-JEPA: if their code supports multi-view input, provide it; otherwise label as "single-view baseline" in tables

### Particle / Slot Count Matching

| Model | Slots/Particles | Notes |
|-------|----------------|-------|
| LPWM variants (A0–A6) | K=30 | Fixed |
| C-JEPA | Match object count | C-JEPA uses flexible object count; use same K as GT objects |
| HCLSM | Slot count from paper | Report their slot count explicitly |
| V-JEPA 2-AC | Patch tokens | Report token count for transparency |

---

## 7. Ablation: GNN Interaction Context Depth

Independent study to answer: how many GNN layers and which edge features matter?

| Variant | GNN Layers | Edge Features | Expected |
|---------|-----------|--------------|---------|
| GNN-0 (baseline) | 0 | — | Worst |
| GNN-1-pos | 1 | 3D relative position only | Position-only interaction |
| GNN-1-full | 1 | concat(z_i, z_j, Δp_3D) | Full 1-hop interaction |
| GNN-2-full | 2 | concat(z_i, z_j, Δp_3D) | 2-hop interaction |
| GNN-global | 1 + global attn | — | GNN + global fallback |

Run on Balls Occlusion (fast, 64×64) for this sub-study. Select best variant for main experiments.

---

## 8. Ablation: Mamba3 State Dimension

| d_state | Rollout MSE (T=20) | FLOPs | Memory |
|---------|--------------------|-------|--------|
| 16 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |
| 128 | — | — | — |

Run on Sketchy. Expected: diminishing returns after d_state=64 for K=30 particle dynamics.

---

## 9. Ablation: lambda_rec Annealing Schedule

Test sensitivity to JEPA annealing rate (RQ: is gradual annealing necessary or can we drop reconstruction immediately?):

| Schedule | Description |
|----------|-------------|
| Immediate | lambda_rec = 0 from epoch 0 |
| Fast (proposed fast) | Anneal to 0 by epoch 50 |
| Standard (proposed) | Anneal to 0 by epoch 80 |
| Slow | Anneal to 0 by epoch 120 |
| Never | lambda_rec = 0.1 throughout |

Monitor: cross-view loss stability, ARI, particle specialization. Collapse detection: if cross-view loss > 2× initial value, flag as collapsed.

---

## 10. Reproducibility Checklist

- [ ] All baselines retrained on our multi-view datasets (no paper numbers used directly)
- [ ] 3 random seeds for all main results; report mean ± std
- [ ] Parameter counts reported for all variants
- [ ] Training curves (loss, ARI, cross-view loss) logged and archived
- [ ] Qualitative particle trajectory videos for all variants at T=20
- [ ] Code released with config files for all variants (following LPWM baseline wrapper pattern)
- [ ] HCLSM reproduced from rightnow-ai/hclsm with our data
- [ ] C-JEPA reproduced from galilai-group/cjepa with our data

---

## 11. Compute Estimate

| Component | GPU-hours (est.) |
|-----------|-----------------|
| A0–A5 ablations × 3 seeds (Sketchy) | 6 variants × 150 epochs × 3 seeds × ~2h/epoch ≈ 5,400 GPU-h |
| A6 main model (Sketchy + CLEVRER + PushT) | 3 datasets × 3 seeds × ~2.5h/epoch × 150 = 3,375 GPU-h |
| HCLSM reproduction | ~1,000 GPU-h |
| C-JEPA reproduction | ~800 GPU-h |
| GNN depth ablation (Balls, fast) | ~200 GPU-h |
| d_state sweep | ~300 GPU-h |
| Annealing schedule study | ~500 GPU-h |
| **Total estimate** | **~11,500 GPU-h** |

Reduce by 40% if fast-ablation dataset (Balls, 64×64) is used for RQ2/RQ3 with full validation only on Sketchy.

---

## 12. References

- LPWM: Tal Daniel et al., 2025
- C-JEPA: Nam, LeCun, Balestriero et al., arXiv 2602.11389 (2026-02)
- HCLSM: Jaber & Jaber, arXiv 2603.29090 (2026-03)
- HWM: Zhang, Assran, Balestriero, Bardes et al., arXiv 2604.03208 (2026-04)
- V-JEPA 2: Meta AI, 2025
- Mamba3: Dao & Gu, "Transformers are SSMs", 2024
