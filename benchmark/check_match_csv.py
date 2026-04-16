import pandas as pd
df = pd.read_csv(r"F:\programme\nxy\Cell registration\benchmark\results\full_registration\A2-1\A2-B-1_to_A2-A-1\registration_matches.csv")
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3).to_string())
