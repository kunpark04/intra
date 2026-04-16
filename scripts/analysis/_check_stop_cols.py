"""Quick remote check: does combo_features.parquet contain stop_fixed_pts_resolved?"""
import pandas as pd
df = pd.read_parquet("/root/intra/data/ml/lgbm_results_v2filtered/combo_features.parquet")
stop_cols = [c for c in df.columns if "stop" in c.lower()]
print(f"stop-related columns: {stop_cols}")
if "stop_fixed_pts_resolved" in df.columns:
    print(f"stop_fixed_pts_resolved sample:")
    print(df["stop_fixed_pts_resolved"].head(10).to_list())
    print(f"non-null count: {df['stop_fixed_pts_resolved'].notna().sum()} / {len(df)}")
else:
    print("stop_fixed_pts_resolved NOT in parquet")
