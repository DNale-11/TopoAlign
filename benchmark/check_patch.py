import pandas as pd
for e in ['patch_0','patch_2','patch_3','patch_5']:
    df = pd.read_csv(rf'f:\programme\nxy\Cell registration\benchmark\results\{e}\summary.csv')
    print(f'{e}: Mean F1={df["match_f1"].mean():.4f}, Median F1={df["match_f1"].median():.4f}')
