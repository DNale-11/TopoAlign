"""Statistical significance tests for ablation experiments."""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_ROOT = Path(r"f:\programme\nxy\Cell registration\benchmark\results")

# Load baseline
baseline = pd.read_csv(RESULTS_ROOT / "tps+rigid" / "summary.csv")
baseline_f1 = baseline.sort_values(["case_id", "moving_name"])["match_f1"].values

experiments = {
    # Ablation studies
    "ablation_rigid_only": "Rigid Only",
    "ablation_no_retry": "No Retry",
    "ablation_no_guided_rematch": "No Guided Rematch",
    "ablation_no_consensus": "No Consensus",
    "ablation_tps_only": "TPS Only",
    # top_k sweep
    "topk_016": "top_k=16",
    "topk_064": "top_k=64",
    "topk_128": "top_k=128",
    "topk_160": "top_k=160",
    "topk_640": "top_k=640",
    # window sweep
    "window_050": "window=50px",
    "window_200": "window=200px",
    # patch sweep
    "patch_0": "patch=0",
    "patch_2": "patch=2x2",
    "patch_3": "patch=3x3",
    "patch_5": "patch=5x5",
}

print(f"{'Experiment':<30} {'Mean F1':>10} {'Delta':>10} {'Wilcoxon p':>12} {'Paired-t p':>12} {'Sig?':>6}")
print("-" * 82)
print(f"{'tps+rigid (baseline)':<30} {baseline_f1.mean():>10.4f} {'---':>10} {'---':>12} {'---':>12} {'---':>6}")

for exp_name, label in experiments.items():
    csv_path = RESULTS_ROOT / exp_name / "summary.csv"
    if not csv_path.exists():
        print(f"{label:<30} {'MISSING':>10}")
        continue
    
    df = pd.read_csv(csv_path)
    exp_f1 = df.sort_values(["case_id", "moving_name"])["match_f1"].values
    
    if len(exp_f1) != len(baseline_f1):
        print(f"{label:<30} {'SIZE MISMATCH':>10}")
        continue
    
    mean_f1 = exp_f1.mean()
    delta = mean_f1 - baseline_f1.mean()
    
    # Wilcoxon signed-rank test (non-parametric, paired)
    try:
        w_stat, w_p = stats.wilcoxon(baseline_f1, exp_f1, alternative='two-sided')
    except:
        w_p = float('nan')
    
    # Paired t-test
    t_stat, t_p = stats.ttest_rel(baseline_f1, exp_f1)
    
    sig = "***" if min(w_p, t_p) < 0.001 else "**" if min(w_p, t_p) < 0.01 else "*" if min(w_p, t_p) < 0.05 else "ns"
    
    print(f"{label:<30} {mean_f1:>10.4f} {delta:>+10.4f} {w_p:>12.2e} {t_p:>12.2e} {sig:>6}")
