import pandas as pd

df = pd.read_csv(r"benchmark/results/synthetic_tps/summary.csv")
rigid = df[df["case_id"].str.startswith("rigid")].sort_values("match_f1")
cols = ["case_id", "match_f1", "match_precision", "match_recall",
        "tps_control_points", "warp_mode", "target_cells", "moving_cells"]
print(rigid[cols].head(20).to_string(index=False))
