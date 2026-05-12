# bcae

Behavior cloning with action-space autoencoders, on a 2D pick-and-place env with nonholonomic dynamics. Tests whether pretraining an autoencoder on 16-step (Δx, Δy) action chunks and supervising the BC policy on the latent beats predicting the raw chunk.

## Env

- Unicycle agent with no reverse (`agent_speed ≥ 0`) and a minimum turning radius (`|ω| ≤ |v| / R_MIN`). The feasible trajectory set becomes Dubins-like (smooth, forward-only).
- 2D arena, pick an object, deliver it inside a goal radius for a few consecutive steps.
- Scripted expert (`expert.py`) hits 100% with two geometric heuristics: opposite-loop escape when the target is inside the natural turn circle, and a wall-trap escape that aims at the arena center when wedged against a wall.

## Pipeline

1. **Collect** demos with the scripted expert → `dataset.npz` (`obs`, `act`, `episode`).
2. **Pretrain AE** on 16-step (Δx, Δy) windows of the demos. `--beta 0` = deterministic AE; `> 0` = VAE.
3. **Train BC**:
   - *delta*: MLP(obs-history) → flat 32-d chunk.
   - *latent*: MLP(obs-history) → z, supervised on the frozen AE encoder's μ (no backprop through the decoder). At rollout, the frozen decoder turns z back into the chunk.
4. **Rollout**: feedback-linearization tracker (`eval_bc.track`) converts the chunk to (thrust, yaw); re-plan every `--execute` env steps.

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
