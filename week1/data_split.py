# week1/data_split.py
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

ds = load_dataset("natealberti/ChestX-Det")

def get_label(pil_mask):
    arr = np.array(pil_mask)
    return 1.0 if arr.sum() > 0 else 0.0

print("=== 라벨 수집 중... ===")
train_indices, train_labels = [], []
for i in range(len(ds['train'])):
    train_indices.append(i)
    train_labels.append(int(get_label(ds['train'][i]['mask'])))

test_indices, test_labels = [], []
for i in range(len(ds['test'])):
    test_indices.append(i)
    test_labels.append(int(get_label(ds['test'][i]['mask'])))

all_indices = [('train', i) for i in train_indices] + [('test', i) for i in test_indices]
all_labels = train_labels + test_labels

print(f"전체: {len(all_indices)}장 | 정상: {all_labels.count(0)} | 이상: {all_labels.count(1)}")

train_data, temp_data, train_lbl, temp_lbl = train_test_split(
    all_indices, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
val_data, test_data, val_lbl, test_lbl = train_test_split(
    temp_data, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

print(f"Train: {len(train_data)}장 | Val: {len(val_data)}장 | Test: {len(test_data)}장")

split = {
    'train': [{'split': s, 'idx': i, 'label': l} for (s,i), l in zip(train_data, train_lbl)],
    'val':   [{'split': s, 'idx': i, 'label': l} for (s,i), l in zip(val_data, val_lbl)],
    'test':  [{'split': s, 'idx': i, 'label': l} for (s,i), l in zip(test_data, test_lbl)],
}

os.makedirs('data', exist_ok=True)
with open('data/data_split.json', 'w') as f:
    json.dump(split, f)

print("data/data_split.json 저장됨")