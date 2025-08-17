#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import pandas as pd

RUN_RE = re.compile(r"sl(?P<seq>\d+)_ (?P<prec>fp\d+)_seed(?P<seed>\d+)")

# More robust: parse from folder name created by launch_ddp.sh
META_RE = re.compile(r"sl(?P<seq>\d+)_ (?P<prec>fp\d+)_seed(?P<seed>\d+)")

def load_run_dir(run_dir: Path):
    # Try metrics.csv first
    csv = run_dir / "metrics.csv"
    jl = run_dir / "metrics.jsonl"
    df = None
    if csv.exists():
        df = pd.read_csv(csv)
    elif jl.exists():
        rows = []
        with open(jl, "r") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if rows:
            df = pd.DataFrame(rows)
    if df is None or df.empty:
        return None

    # Basic aggregations (adjust cols to your schema)
    # Expect columns: step, loss, samples_per_s, tokens_per_s, gpu_util
    agg = {
        "steps": df["step"].max() if "step" in df else len(df),
        "loss_last": df["loss"].dropna().iloc[-1] if "loss" in df and not df["loss"].dropna().empty else None,
        "samples_per_s_mean": df.get("samples_per_s", pd.Series(dtype=float)).mean(),
        "tokens_per_s_mean": df.get("tokens_per_s", pd.Series(dtype=float)).mean(),
        "gpu_util_mean": df.get("gpu_util", pd.Series(dtype=float)).mean(),
    }

    # Extract seq_len/precision/seed from folder name
    m = re.search(r"sl(?P<seq>\d+)_([a-z]+)?(?P<prec>fp\d+)_seed(?P<seed>\d+)", run_dir.name)
    if m:
        agg.update({
            "seq_len": int(m.group("seq")),
            "precision": m.group("prec"),
            "seed": int(m.group("seed")),
        })
    return agg


def main(root: str, out_csv: str):
    root_path = Path(root)
    run_dirs = list(root_path.glob("**/experiments/logs/*"))
    rows = []
    for d in run_dirs:
        if d.is_dir():
            agg = load_run_dir(d)
            if agg:
                agg["run_dir"] = str(d)
                rows.append(agg)
    if not rows:
        print("No metrics found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Grouped summary and speedups
    grp = df.groupby(["seq_len", "precision"]) ["tokens_per_s_mean"].agg(["count","mean","median"]).reset_index()

    # Speedup vs fp32 per seq_len
    def speedup_block(g):
        base = g.loc[g["precision"]=="fp32", "mean"]
        base = float(base.iloc[0]) if not base.empty else None
        g = g.copy()
        g["speedup_vs_fp32"] = g["mean"].apply(lambda x: (x/base) if (base and base>0) else None)
        return g

    summary = grp.groupby("seq_len", group_keys=False).apply(speedup_block)

    summary_path = Path(out_csv).with_name("summary_" + Path(out_csv).name)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote:\n - {out_csv}\n - {summary_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root or parent of experiments/")
    ap.add_argument("--out", default="combined_metrics.csv")
    args = ap.parse_args()
    main(args.root, args.out)