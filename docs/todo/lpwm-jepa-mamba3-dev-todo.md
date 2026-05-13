# TODO: LPWM-JEPA-Mamba3 Development

Related: `docs/design/lpwm-jepa-mamba3-dynamics.md`

---

## Prerequisites (Block all Phase 2 work)

- [ ] 3D-LPWM (Transformer) baseline validated
  - [ ] Cross-view loss demonstrably improves depth estimation
  - [ ] Quantitative depth rank consistency improvement confirmed
- [ ] Particle token format confirmed compatible with Mamba3 input (same d_model=512)

---

## Phase 2a: JEPA Training Objective

### Implementation

- [ ] `EMAEncoder`: copy of DLPEncoder, updated with exponential moving average (tau=0.996)
  - [ ] Implement EMA weight update step
  - [ ] Ensure stop-gradient on target encoder outputs
- [ ] `L_jepa` loss: `||z_hat_{t+1} - sg(z_bar_{t+1})||^2`
  - [ ] Add to training loop alongside existing losses
- [ ] `lambda_rec` annealing scheduler
  - [ ] Epoch 0-30: lambda_rec = 1.0
  - [ ] Epoch 30-80: linear 1.0 -> 0.1
  - [ ] Epoch 80+: lambda_rec = 0.0
- [ ] Loss composition: `L = L_jepa + lambda_cv * L_cv + beta_kl * KL + lambda_rec(t) * L_rec`

### Validation

- [ ] Collapse detection: monitor cross-view loss stability during JEPA training
- [ ] Verify 4 collapse prevention mechanisms active:
  - [ ] EMA stop-gradient blocks trivial gradient path
  - [ ] Cross-view loss maintains I(z; S_3D) > 0
  - [ ] KL regularization prevents distribution collapse
  - [ ] Multi-particle combinatorial structure provides implicit protection
- [ ] Reconstruction probe: track I(z; x) to ensure representations remain informative

---

## Phase 2b: GNN-Mamba3 Dynamics

### GNN (Lightweight MPNN)

- [ ] Edge message MLP: `phi_e(z_i, z_j, ||p_i - p_j||_3D)` -> R^{d_e=128}
- [ ] Attentional aggregation: `c_i = sum_j alpha_ij * m_ij`
  - [ ] Attention weights: `alpha_ij = softmax_j(psi(m_ij))`
- [ ] Support 1-2 layer stacking
- [ ] Edge features use relative 3D position (translation equivariant)
- [ ] Test: verify output shape (K, d_c) for K=30 particles

### Mamba3 Temporal Block

- [ ] Select library: `mamba-ssm` or `mamba3-torch`
- [ ] Input: `concat(z_t^{(i)}, c_t^{(i)})` in R^{d + d_c}
- [ ] Configure: d_state=64, d_conv=4, expand=2
- [ ] Verify O(1) per-step inference (recurrence mode)
- [ ] Verify input-dependent A_t, B_t selectivity mechanism

### GNNMamba3DynamicsBlock (Interleaved)

- [ ] Implement interleaved structure:
  - [ ] Layer 1: GNN(z_t) -> c_t (spatial interaction context)
  - [ ] Layer 2: Mamba3([z_t; c_t]) -> z_tilde_t (temporal evolution)
  - [ ] Layer 3: GNN(z_tilde_t) -> c_tilde_t (recompute interactions)
  - [ ] Layer 4: Mamba3([z_tilde; c_tilde]) -> z_hat_{t+1}
- [ ] Stack N=3 interleaved blocks
- [ ] Match hidden state dimensions to pint_dim=512
- [ ] Total dynamics params target: ~12M (GNN ~3M + Mamba3 ~9M)

### Integration

- [ ] Replace `ParticleSpatioTemporalTransformer` in `DLPDynamics`
- [ ] Ensure backward compatibility with existing DLPContext (4-layer Tx, 8H, dim=512)
- [ ] DLPDecoder remains unchanged (only used during lambda_rec > 0 phase)
- [ ] Verify training loop: forward pass + loss computation + backward pass correct

### Selective Gating Verification

- [ ] Test semantic behavior:
  - [ ] No interaction -> A_t ~ 1 (inertial memory preserved)
  - [ ] Strong collision -> A_t ~ 0, B_t large (memory overwritten)
  - [ ] Occlusion -> relevant dims A_t ~ 1 (preserve until reappearance)
- [ ] Visualize gating values on toy collision scenarios

---

## Phase 2c: Integration and Ablation Runs

- [ ] Ablation A: Mamba3 without GNN (pure particle-state temporal)
- [ ] Ablation B: GNN + Mamba3 (full proposed)
- [ ] Ablation C: GNN + Transformer temporal (isolate Mamba3 contribution)
- [ ] Metrics to collect: multi-step rollout FVD, per-particle tracking, long-horizon stability

---

## Risk Mitigation Checkpoints

| Checkpoint | Condition | Action if Failed |
|-----------|-----------|-----------------|
| JEPA collapse | cross-view loss > 2x initial | Halt annealing, increase lambda_cv |
| Mamba3 d_h insufficient | Short-horizon MSE > Transformer by >10% | Sweep d_h up to 128; add spatial attention fallback |
| EMA instability | Training loss diverges after EMA warmup | Separate warmup phases; fix EMA during L_cv burn-in |
| GNN overhead | Latency increase >2x with no metric gain | Gate GNN; bypass for low-interaction regimes |

---

## Code Organization (Implemented)

```
module/dynamic/
├── __init__.py                # Package exports (lazy import for dynamics)
├── config.py                  # GNNMamba3Config dataclass
├── mpnn.py                    # LightweightMPNN (2-layer MPNN, 1.86M params)
├── mamba3_temporal.py         # Mamba3TemporalBlock (PyTorch fallback + mamba-ssm)
├── gnn_mamba3_block.py        # GNNMamba3Block + GNNMamba3Transformer
├── gnn_mamba3_dynamics.py     # GNNMamba3Dynamics (drop-in DLPDynamics replacement)
├── ema_encoder.py             # EMAEncoder (wraps any encoder with EMA)
└── jepa_loss.py               # JEPALoss + AnnealingScheduler + CombinedLoss
```

### Verified (2026-05-12)

- Forward pass: B=2, T=19, K=30 → correct output shapes
- Autoregressive sampling: 5 cond frames + 3 steps → T=8
- Gradient flow: verified back to input through GNN-Mamba3
- EMA: stop-gradient verified, decay update working
- JEPA loss: gradient flows through predictor only
- Output dict: all 16 keys matching DLPDynamics interface
- Total params: 21.57M (GNNMamba3Dynamics with projection+decoder)
- SSM backend: PyTorch fallback active (mamba-ssm Mamba3 optional)

