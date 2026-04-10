import shutil
from pathlib import Path

source_dir = Path(r"F:\programme\nxy\Cell registration\benchmark\icp_results")
base_output_dir = Path(r"F:\programme\nxy\Cell registration\benchmark\icp_exported_artifacts")

# 创建两个独立的文件夹
mask_output_dir = base_output_dir / "register mask文件"
feature_output_dir = base_output_dir / "registrer feature文件"

mask_output_dir.mkdir(parents=True, exist_ok=True)
feature_output_dir.mkdir(parents=True, exist_ok=True)

copied_masks = 0
copied_features = 0

print(f"Scanning directory: {source_dir}")

for case_dir in source_dir.iterdir():
    if not case_dir.is_dir():
        continue
        
    for pair_dir in case_dir.iterdir():
        if not pair_dir.is_dir():
            continue
            
        pair_name = pair_dir.name
        
        mask_path = pair_dir / "registered_mask.tif"
        feature_path = pair_dir / "registered_features.csv"
        
        if mask_path.exists():
            # 命名为 <pair_name>.tif
            dest_mask = mask_output_dir / f"{pair_name}.tif"
            shutil.copy2(mask_path, dest_mask)
            copied_masks += 1
            
        if feature_path.exists():
            # 命名为 <pair_name>.csv
            dest_feature = feature_output_dir / f"{pair_name}.csv"
            shutil.copy2(feature_path, dest_feature)
            copied_features += 1

print(f"Done!")
print(f"Masks exported to: {mask_output_dir} ({copied_masks} files)")
print(f"Features exported to: {feature_output_dir} ({copied_features} files)")
