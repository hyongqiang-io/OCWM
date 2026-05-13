# LPWM Phase 2: Multi-View JEPA + GNN-Mamba3 Dynamics

## Status

- Stage: design (pending 3D-LPWM baseline validation)
- Predecessor: `lpwm-baseline.md` (3D-LPWM with cross-view loss)
- Decision date: 2026-05-11

## Overview

Replace the Transformer-based spatio-temporal dynamics in DLPDynamics with:
1. **Multi-view training** for 3D-grounded particle representations
2. **JEPA latent prediction** (drop pixel reconstruction dependency)
3. **GNN + Mamba3 interleaved blocks** for temporal dynamics with interaction-aware memory

## Motivation

The current DLPDynamics uses a `ParticleSpatioTemporalTransformer` that alternates:
- Spatial self-attention: O(K^2), K~25 particles (cheap)
- Temporal causal attention: O(T^2), T=horizon (expensive at rollout)

Additionally, the pixel reconstruction loss forces the latent space to retain rendering details irrelevant to physics. The proposed architecture addresses both issues.

---

## 1. Multi-View Training: Information-Theoretic Justification

### Formulation

Given views V_1, V_2 of the same scene and latent scene structure S:

```
I(z; S) >= I(z; V_2 | V_1)
```

Cross-view reconstruction loss maximizes I(z; V_2 | V_1), forcing particles to encode view-invariant 3D geometry.

### Information Gain

Single-view mutual information I_1 = I(z; V_1). Dual-view provides:

```
I(z; S) <= I(z; V_1, V_2) = I_1 + I(z; V_2 | V_1, z)
```

The cross-view loss in `DLP3D.compute_cross_view_loss` implements:

```
L_cv = ||Dec(T_{1->2}(Unproj(z_2D, z_depth))) - x_{V_2}||^2
```

This is a geometric equivariance constraint: the representation space must be compatible with SE(3) transformations. Strictly stronger than data augmentation.

### Depth Ambiguity

- 2 views: residual convex/concave ambiguity
- >=3 views: fully resolves depth under generic camera placement

---

## 2. JEPA Latent Prediction

### Loss Function

```
L_jepa = ||z_hat_{t+1} - sg(z_bar_{t+1})||^2
z_bar = EMA_encoder(x_{t+1})
```

### Why Not Pixel Reconstruction

Pixel reconstruction requires:
```
Rate = I(z; x) >= I(z; S_geometry) + I(z; x | S_geometry)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^
                                      irrelevant rendering details
```

JEPA's EMA target encoder automatically discards information not useful for next-step prediction, achieving:
```
min_theta  I(z; x) - beta * I(z; z_{t+1})
```

### Collapse Prevention (4 mechanisms)

| Mechanism | Type | Sufficiency |
|-----------|------|-------------|
| EMA target (stop-gradient) | Blocks trivial gradient path | Necessary, not sufficient alone |
| Cross-view loss | Guarantees I(z; S_3D) > 0 | Strong regularizer (unique to this arch) |
| KL regularization | Prevents distribution collapse | Auxiliary |
| Multi-particle combinatorial structure | Exponential mapping space | Implicit protection |

Key advantage over standard V-JEPA: cross-view loss provides an information lower bound. Even if the JEPA predictor attempts to collapse, cross-view reconstruction demands I(z; S) >= H(V_2|V_1) > 0.

### Training Schedule

```
L = L_jepa + L_cross_view + KL_reg + lambda_rec * L_rec
```

Anneal lambda_rec from 1.0 -> 0 over training. Cross-view loss remains as the geometric anchor throughout.

---

## 3. GNN-Mamba3 Interleaved Dynamics

### Problem with Current Architecture

The `SpatioTemporalBlock` computes particle interactions independently at each timestep. Interaction patterns that evolve over time (approaching collision, gravitational pairing changes, occlusion/reconnection) must be inferred implicitly from particle state history. No explicit interaction memory exists.

### Proposed Architecture: Interleaved GNN-Mamba3 Blocks

```
Layer 1: GNN(z_t)        -> c_t^{(i)}        [spatial interaction, produces per-node context]
Layer 2: Mamba3([z_t; c_t]) -> z_tilde_t     [temporal evolution, input includes interaction context]
Layer 3: GNN(z_tilde_t)  -> c_tilde_t        [recompute interactions on evolved state]
Layer 4: Mamba3([z_tilde; c_tilde]) -> z_hat_{t+1}
```

Each Mamba3 layer receives both particle state AND GNN-produced interaction context. The hidden state thus encodes joint node+edge dynamics.

### Mathematical Form

```
h_t = A_t * h_{t-1} + B_t * concat(z_t^{(i)}, sum_{j in N(i)} phi(z_t^{(i)}, z_t^{(j)}))
```

Where A_t, B_t are input-dependent (Mamba3 selectivity mechanism):
```
A_t = diag(sigma(Linear(x_t))) in [0,1]^{d_h}
```

### Selective Gating Physical Semantics

- GNN reports "no neighbor interaction" -> A_t -> 1 (maintain inertial memory)
- GNN reports "strong collision event" -> A_t -> 0, B_t large (overwrite with new info)
- Particle occluded -> corresponding dims A_t -> 1 (preserve memory until reappearance)

### GNN Design (Lightweight MPNN)

For K~25 particles, 1-2 layer MPNN is sufficient:

```
m_ij = phi_e(z^{(i)}, z^{(j)}, ||p^{(i)} - p^{(j)}||)    # edge message
c^{(i)} = sum_j alpha_ij * m_ij                             # attentional aggregation
alpha_ij = softmax_j(psi(m_ij))                             # attention weights
```

Edge features use relative 3D position -> translation equivariant, directly compatible with 3D particle coordinates.

### Complexity Comparison

| Property | Temporal Transformer | Mamba3 |
|----------|---------------------|--------|
| Training complexity | O(T^2 * d) | O(T * d_h) linear scan |
| Inference latency (per step) | O(T) with KV cache | O(1) recurrence |
| Long-range dependency | Full T-step attention | Bounded by d_h |
| Causality | Requires mask | Natural |
| Selective forgetting | No explicit mechanism | Input-dependent gating |

### Alternative Topologies Considered

**A. Dual-stream parallel Mamba3 (node stream + edge stream)**

```
[h_t^n]   [A^n    W^{e->n}] [h_{t-1}^n]   [B^n * z_t]
[h_t^e] = [W^{n->e}  A^e  ] [h_{t-1}^e] + [B^e * e_t]
```

Rejected: introduces extra parameters without clear benefit. Node/edge information is highly redundant (interactions are determined by states), so a shared hidden state is more parameter-efficient.

**B. Interleaved GNN-Mamba3 (SELECTED)**

Reasons:
1. Information efficiency: z and c redundancy naturally absorbed by shared hidden state
2. Code compatibility: current SpatioTemporalBlock is already "Spatial -> Temporal" interleaving
3. Physical semantics of selective gating are clear with interaction-aware input

**C. Graph-Mamba (graph structure embedded in SSM transition)**

```
A_t^{(i)} = sigma(Linear(z_t^{(i)}) + sum_j alpha_ij * Linear(z_t^{(j)}))
```

Rejected: conflates spatial and temporal axes. For particle systems these are semantically distinct and should remain separable for interpretability.

---

## 4. Full Architecture Data Flow

```
x_t^{V1}, x_t^{V2}
    |
    v
Encoder (shared) ──────────────────────────────────> EMA Encoder ──> z_bar_{t+1}
    |                                                                    |
    v                                                                    |
z_t^{(i)} = (z_pos, z_depth, z_scale, z_features, z_obj_on)            |
    |                                                                    |
    v                                                                    |
┌─────────────────────────────────────────┐                             |
│  GNN-Mamba3 Block x N_layers            │                             |
│                                         │                             |
│  GNN: z_t -> c_t (interaction context)  │                             |
│  Mamba3: [z_t; c_t] -> z_hat_{t+1}     │                             |
└─────────────────────────────────────────┘                             |
    |                                                                    |
    v                                                                    v
z_hat_{t+1} ─────────────────────── L_jepa = ||z_hat - sg(z_bar)||^2
    |
    v (during training, for annealing only)
Cross-view render -> L_cv = ||render(T * unproj(z)) - x_{V2}||^2
```

### Training Losses

```
L_total = L_jepa + lambda_cv * L_cross_view + beta_kl * L_KL + lambda_rec * L_rec(t)

where lambda_rec(t) anneals: 1.0 -> 0 over training epochs
```

---

## 5. Capability Bounds

### Hard Limits

| Factor | Source | Quantification |
|--------|--------|----------------|
| Particle count K | Scene complexity ceiling | Max K independent objects |
| Mamba state dim d_h | Long-range memory capacity | Cannot precisely recall >d_h independent single-occurrence events |
| EMA decay tau | Target staleness | Too fast -> collapse; too slow -> target lags |
| View count | 3D constraint quality | 2 views = depth ambiguity; >=3 resolves |
| GNN depth (layers) | Spatial reasoning radius | L-layer MPNN = L-hop neighborhood |

### Soft Limits (Addressable by Design)

1. **Non-rigid deformation**: Perspective projection assumes rigid points. Non-rigid requires encoding local deformation in z_features, increasing d requirement.

2. **Topology changes** (object appear/disappear): z_obj_on Beta distribution can soft-kill particles, but new object appearance needs a "free particle pool" mechanism. Fixed K is limiting.

3. **Causal inference**: JEPA learns correlation (next-state prediction), not causal structure. Counterfactual reasoning ("what if force direction changed") requires additional action-conditioned disentanglement. The existing action_condition mechanism provides a hook but Mamba3 hidden state disentanglement is unverified.

4. **Multi-modal futures**: L_jepa = ||z_hat - z*||^2 is unimodal. If futures branch (ball hitting a fork), MSE learns the mean. Solutions: generative predictor (VQ, flow matching) or rely on z_ctx sampling to disambiguate modes.

### Capability Gains from GNN Interaction Context

| Capability | Before (SpatialAttn + Mamba3) | After (GNN + Mamba3) |
|------------|------|------|
| Collision prediction | Mamba3 infers from position trends | GNN directly encodes proximity, Mamba3 extrapolates |
| Multi-body gravity | Independent per-step computation, no acceleration history | Force field history in Mamba3 state predicts orbit curvature |
| Post-occlusion recovery | Depends on Mamba3 d_h capacity alone | Interaction context provides "last interaction before occlusion" cue |
| New upper bound | -- | GNN depth limits multi-hop reasoning (2-hop for 2-layer) |

For K=25 particles, a 2-layer GNN covers nearly all pairwise paths (graph diameter <= 4 typically), so this is not a practical bottleneck.

### Information Efficiency Advantage

Compared to pixel-level world models (IRIS, Genie):

```
Useful bits for control     I(z; S_physics)
─────────────────────── = ─────────────────── -> 1.0
Total bits processed        I(z; x_pixels)
```

The architecture actively discards rendering details irrelevant to physics via JEPA + multi-view, yielding:
- Smaller models learn better dynamics
- Rollout errors not amplified by pixel noise
- Higher sample efficiency (denser useful gradient signal per frame)

---

## 6. Implementation Plan

### Prerequisites

- [ ] 3D-LPWM (Transformer) validated: cross-view loss demonstrably improves depth estimation
- [ ] Particle token format confirmed compatible with Mamba3 input (same d_model)

### Phase 2a: JEPA Training Objective

1. Add EMA target encoder (copy of DLPEncoder, updated with momentum)
2. Add L_jepa loss alongside L_rec
3. Implement lambda_rec annealing schedule
4. Verify no representation collapse via cross-view loss stability

### Phase 2b: GNN-Mamba3 Dynamics

1. Implement lightweight MPNN module (edge MLP + attentional aggregation)
2. Implement Mamba3 temporal block (using `mamba-ssm` or `mamba3-torch`)
3. Create `GNNMamba3DynamicsBlock` with interleaved structure
4. Replace `ParticleSpatioTemporalTransformer` in `DLPDynamics`
5. Match hidden state dimensions to existing pint_dim=512

### Phase 2c: Integration and Ablation

- Ablation A: Mamba3 without GNN context (pure particle-state temporal)
- Ablation B: GNN + Mamba3 (full proposed architecture)
- Ablation C: GNN + Transformer temporal (to isolate Mamba3's contribution)
- Metric: multi-step rollout FVD, per-particle tracking accuracy, long-horizon stability

---

## 7. Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mamba3 d_h insufficient for Transformer-equivalent prediction | Degraded short-horizon accuracy | Sweep d_h; keep spatial attention fallback layer |
| EMA decay / cross-view coupling instability | Training divergence | Separate warmup phases; fix EMA during L_cv burn-in |
| lambda_rec annealing too aggressive | Collapse before JEPA signal stabilizes | Monitor I(z; x) via reconstruction probe; halt annealing if probe degrades |
| GNN adds latency without benefit for simple scenes | Overhead on easy tasks | Gate GNN contribution; bypass for low-interaction regimes |

---

## References

- Mamba3: Dao & Gu, 2025
- V-JEPA: Bardes et al., 2024 (video prediction in latent space)
- LPWM: Tal Daniel et al., 2025
- Graph-Mamba: Wang et al., 2024
- Universal Approximation for SSMs: Orvieto et al., 2023
- Information Bottleneck: Tishby et al., 2000
