import pandas as pd
df = pd.read_csv(r"F:\programme\nxy\Cell registration\benchmark\results\full_registration\summary.csv")
print(f"Mean F1:   {df['match_f1'].mean():.4f}")
print(f"Median F1: {df['match_f1'].median():.4f}")
print(f"Std F1:    {df['match_f1'].std():.4f}")
print(f"Count:     {len(df)}")
# Find pair name column
name_cols = [c for c in df.columns if 'pair' in c.lower() or 'name' in c.lower() or 'case' in c.lower() or 'moving' in c.lower()]
print(f"\nName columns: {name_cols}")
if name_cols:
    nc = name_cols[0]
    print(f"\nWorst 10 (by {nc}):")
    print(df.nsmallest(10, 'match_f1')[[nc, 'match_f1']].to_string(index=False))
    print(f"\nBest 10:")
    print(df.nlargest(10, 'match_f1')[[nc, 'match_f1']].to_string(index=False))
else:
    print("\nColumns:", list(df.columns[:20]))
