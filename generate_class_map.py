import pandas as pd
import json
import os

# CHANGE THESE PATHS to your actual CSVs
csv_paths = [
    "/kaggle/input/final_dataset/train_clip_results.csv",
    "/kaggle/input/final_dataset/val_clip_results.csv",
    "/kaggle/input/final_dataset/test_clip_results.csv",
]

dfs = []
for p in csv_paths:
    if os.path.exists(p):
        dfs.append(pd.read_csv(p))

if not dfs:
    raise RuntimeError("No CSV files found. Fix paths in generate_class_map.py")

df = pd.concat(dfs, ignore_index=True)

possible_cols = ["Class", "class", "label", "Label", "target", "category"]
label_col = None

for c in possible_cols:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    for c in df.columns:
        if c not in ["file_name", "filename", "image"]:
            label_col = c
            break

if label_col is None:
    raise RuntimeError("Could not detect label column.")

classes = sorted(df[label_col].unique())
idx2class = {i: cls for i, cls in enumerate(classes)}

with open("class_idx_to_name.json", "w") as f:
    json.dump(idx2class, f, indent=4)

print("Saved class_idx_to_name.json →", idx2class)
