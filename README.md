# bcae

Behavior cloning with action-space autoencoders, on a 2D pick-and-place env. Tests whether pretraining an autoencoder on action chunks of the end-effector trajectory and supervising the BC policy on the latent beats predicting the raw chunk.

## Env

- Planar N-link revolute arm (default N=3), no gravity. Base at the arena center, total reach 1.0, joint inertia + light damping. Action = per-joint torque in `[-1, 1]`.
- Pick an object with the EE and deliver it inside a goal radius for a few consecutive steps.
- Configurable: `PlanarArmEnv(n_joints=N)`. Obs is `[ee_x, ee_y, cos θ_N, sin θ_N, q, qd, object, goal, attached]` (`2N + 9` dims).
- Scripted expert: operational-space PD on the EE via the transpose Jacobian (`τ = Jᵀ·F_des − K_Q·qd`). 95–100% across N ∈ {2, 3, 4}.

## Pipeline

1. **Collect** demos with the scripted expert → `dataset.npz` (`obs`, `act`, `episode`).
2. **Pretrain AE** on L-step (Δx, Δy) EE-trajectory windows. `--beta 0` = deterministic AE; `> 0` = VAE.
3. **Train BC**:
   - *delta*: MLP(obs-history) → flat 2L-d chunk of EE deltas.
   - *latent*: MLP(obs-history) → z, supervised on the frozen AE encoder's μ (no backprop through the decoder). At rollout, the frozen decoder turns z back into the chunk.
4. **Rollout**: operational-space tracker (`eval_bc.track`) follows the predicted EE waypoints by computing `F_des = K_P·(p_ref − ee) + K_D·(v_ref − ee_vel)` and mapping it to joint torques via `τ = Jᵀ·F_des − K_Q·qd`. Re-plan every `--execute` env steps (default = `max(8, window·3/4)`).

## Layout

```
bcae/
├── env.py, expert.py            sim + scripted expert
├── collect_data.py              write dataset.npz
├── train_vae.py                 AE/VAE on action chunks
├── train_bc.py                  BC MLP; --vae_ckpt enables latent mode
├── eval_bc.py                   eval one ckpt (success rate, optional video/viz)
├── eval_many.py                 eval many ckpts → CSV
├── sweeps/
│   ├── run_sweep.py             4-arch × 6-demo-count β-ablation sweep
│   └── multi_seed_sweep.py      multi-seed sweep, 2-GPU parallel train + eval; --fixed_vae isolates BC-seed variance
├── plots/
│   ├── plot_eval.py             success vs demos, Wilson 95% CIs
│   ├── plot_seeds.py            seed variability (mean ± seed σ + per-seed scatter)
│   └── plot_compare.py          overlay multiple eval CSVs (one panel per arch)
└── tools/
    ├── vae_interp.py            VAE latent interpolation viz
    ├── plot_demos.py            ground-truth demo windows viz
    └── visualize_dataset.py     dataset scatter / overview
```

## Setup

```bash
conda env create -f environment.yml
conda activate bcae
```

## Run

```bash
# One-time
python collect_data.py --episodes 1000

# Single-seed sweep at chosen β (4 archs × 6 demo counts)
python sweeps/run_sweep.py --out_dir sweep         --beta 0.0
python sweeps/run_sweep.py --out_dir sweep_beta1e3 --beta 1e-3 --wandb_suffix=-beta1e3
python sweeps/run_sweep.py --out_dir sweep_beta1   --beta 1.0  --wandb_suffix=-beta1

# Multi-seed sweep (β=0, 5 seeds, parallel across both GPUs)
python sweeps/multi_seed_sweep.py --out_dir sweep_multi --seeds 0 1 2 3 4 --demos 25 100 500

# Same, but isolate BC-seed variance with one fixed VAE
python sweeps/multi_seed_sweep.py --out_dir sweep_bc_only --archs dec_z8 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --fixed_vae sweep_multi/vae_z8_s0.pt

# Post-hoc eval + plot
python eval_many.py --ckpts sweep/bc_*.pt --episodes 1000 --out_csv sweep/eval_1000.csv
python plots/plot_eval.py --csv sweep/eval_1000.csv --title_suffix " — β=0"

# Overlay β values
python plots/plot_compare.py \
  --csvs   sweep/eval_1000.csv sweep_beta1e3/eval_1000.csv sweep_beta1/eval_1000.csv \
  --labels "β=0"               "β=1e-3"                    "β=1" \
  --out compare_betas.png

# Seed plot (from multi-seed CSV)
python plots/plot_seeds.py --csv sweep_multi/eval_100.csv
```

## Notes

- `--wandb_suffix=-foo` needs the `=` (argparse rejects values starting with `-`).
- Step budget (`--max_steps`), not epochs, is fixed across demo counts for fair compute.
- `run_sweep.py` writes per-eval-point viz grids to `sweep*/viz_<tag>/step_*.png`.
- Long jobs: spawn in tmux (`tmux new-session -d -s <name> -c $(pwd) "bash -lc '<cmd>; exec bash'"`).
