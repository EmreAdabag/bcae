import argparse
import csv
import os
import re
import subprocess
import sys


def parse_arch_demos(ckpt_path: str):
    """sweep/bc_<arch>_d<demos>(_s<seed>)?.pt -> ('<arch>', <demos>). Returns (None, None) if no match."""
    name = os.path.basename(ckpt_path)
    m = re.match(r"bc_(.+)_d(\d+)(?:_s\d+)?\.pt$", name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main():
    p = argparse.ArgumentParser(description="Evaluate a set of BC checkpoints at a fixed episode count and write results to CSV.")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--out_csv", type=str, required=True)
    args = p.parse_args()

    rows = []
    for ckpt in args.ckpts:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_bc.py"),
               "--ckpt", ckpt, "--episodes", str(args.episodes)]
        print(f"=== {ckpt} ===", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"!! FAILED rc={r.returncode}: {r.stderr[-500:]}", flush=True)
            continue
        for line in r.stdout.splitlines():
            if line.startswith("Success:"):
                m = re.search(r"Success:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", line)
                if m:
                    num = int(m.group(1)); den = int(m.group(2))
                    arch, demos = parse_arch_demos(ckpt)
                    rows.append({
                        "ckpt": os.path.basename(ckpt),
                        "arch": arch,
                        "demos": demos,
                        "successes": num,
                        "episodes": den,
                        "rate": num / max(den, 1),
                    })
                    print(f"  -> {num}/{den} ({100*num/max(den,1):.1f}%)", flush=True)
                break

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "arch", "demos", "successes", "episodes", "rate"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nSaved {args.out_csv} with {len(rows)} rows")


if __name__ == "__main__":
    main()
