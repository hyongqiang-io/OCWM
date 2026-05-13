# TODO: Experiment Setup

Related: `docs/design/experiment-setup.md`

---

## Phase 0: Data Preparation

- [ ] Sketchy dataset download + multi-view synthetic generation (camera T_{1->2})
- [ ] CLEVRER dataset: synthesize second view from physics engine
- [ ] PushT (OXE): identify real dual-camera subset
- [ ] Balls Occlusion dataset generation (64x64, fast ablation)
- [ ] Verify episode-level train/val/test splits (80/10/10), no frame leakage
- [ ] Standardize data loading pipeline for all 4 datasets (same format)

## Phase 1: Baseline Reproduction

- [ ] A0 (LPWM): reproduce original LPWM results on Sketchy (3 seeds)
- [ ] Verify parameter count ~15M for Transformer dynamics
- [ ] C-JEPA: clone galilai-group/cjepa, adapt to our data format
- [ ] HCLSM: clone rightnow-ai/hclsm, adapt to our data format
- [ ] V-JEPA 2-AC: setup from meta-research/jepa
- [ ] Confirm all baselines train successfully on Balls Occlusion (fast validation)

## Phase 2: Ablation Variants Implementation

- [ ] A1 (LPWM-3D): add cross-view loss to baseline Transformer
- [ ] A2 (LPWM-JEPA): add JEPA loss (no 3D, no Mamba/GNN)
- [ ] A3 (LPWM-3D-JEPA): JEPA + cross-view + Transformer
- [ ] A4 (LPWM-SSM): Mamba3 only (no GNN), match params to ~15M
- [ ] A5 (LPWM-GNN-Tx): GNN + Transformer, GNN ~3M + Tx ~12M
- [ ] A6 (LPWM-GNN-Mamba3): full proposed model

## Phase 3: Sub-Ablations

- [ ] GNN depth ablation (GNN-0 / GNN-1-pos / GNN-1-full / GNN-2-full / GNN-global) on Balls
- [ ] Mamba3 d_state sweep (16/32/64/128) on Sketchy
- [ ] lambda_rec annealing schedule comparison (immediate / fast / standard / slow / never)
- [ ] Collapse detection: monitor cross-view loss > 2x initial threshold

## Phase 4: Main Experiments

- [ ] Run A0-A6 x 3 seeds on Sketchy (150 epochs each)
- [ ] Run A6 x 3 seeds on CLEVRER
- [ ] Run A6 x 3 seeds on PushT
- [ ] Run external baselines (C-JEPA, HCLSM, V-JEPA 2-AC) on all datasets
- [ ] Collect all metrics: MSE, FVD, LPIPS, SSIM at T=5/10/20
- [ ] Collect object-centric metrics: ARI, tracking accuracy, particle specialization
- [ ] Collect 3D metrics: cross-view PSNR/SSIM, depth rank consistency, equivariance error
- [ ] Collect efficiency metrics: FLOPs, latency, throughput, memory

## Phase 5: Downstream Planning Evaluation

- [ ] Integrate HWM planner on top of LPWM-GNN-Mamba3
- [ ] Integrate HWM planner on top of original LPWM
- [ ] Integrate HWM planner on top of V-JEPA 2-AC
- [ ] Run pick-&-place success rate evaluation (100 test episodes)
- [ ] VLA sample efficiency comparison: fine-tune on N={1%,5%,10%,50%,100%} episodes

## Phase 6: Paper Writing

- [ ] Report exact parameter counts for all variants
- [ ] Generate qualitative particle trajectory videos (T=20)
- [ ] Training curves visualization (loss, ARI, cross-view loss)
- [ ] Compile tables for all RQs with mean +/- std
- [ ] Articulate distinction vs HCLSM (dynamic context vs static causal graph)
- [ ] Articulate distinction vs C-JEPA (spatiotemporal 3D vs spatial masking)

## Compute Budget Tracking

| Phase | Estimated GPU-hours | Actual |
|-------|-------------------|--------|
| Baseline reproduction | ~2,000 | — |
| A0-A5 ablations (Sketchy, 3 seeds) | ~5,400 | — |
| A6 main (3 datasets, 3 seeds) | ~3,375 | — |
| Sub-ablations (Balls + Sketchy) | ~1,000 | — |
| External baselines | ~1,800 | — |
| Downstream eval | ~500 | — |
| **Total** | **~11,500** | — |
