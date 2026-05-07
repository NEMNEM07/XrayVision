# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import pydicom
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RSNA_DCM_DIR = 'data/stage_2_train_images'
RSNA_LABELS  = 'data/stage_2_train_labels.csv/stage_2_train_labels.csv'


def load_rsna_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)
    img_pil = Image.fromarray(img).convert('RGB')
    img_pil = img_pil.resize((224, 224), Image.BILINEAR)
    return img_pil


def make_rsna_mask(bboxes, orig_size=1024):
    mask = np.zeros((orig_size, orig_size), dtype=np.float32)
    for x, y, w, h in bboxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        mask[y:y+h, x:x+w] = 1.0
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize((224, 224), Image.NEAREST)
    return np.array(mask_pil).astype(np.float32) / 255.0


def build_rsna_samples():
    labels = pd.read_csv(RSNA_LABELS)
    grouped = labels.groupby('patientId')
    samples = []
    for pid, rows in grouped:
        dcm_path = os.path.join(RSNA_DCM_DIR, f'{pid}.dcm')
        if not os.path.exists(dcm_path):
            continue
        target = int(rows['Target'].iloc[0])
        bboxes = []
        if target == 1:
            bboxes = rows[['x', 'y', 'width', 'height']].dropna().values.tolist()
        samples.append({
            'type': 'rsna',
            'dcm_path': dcm_path,
            'label': float(target),
            'bboxes': bboxes
        })
    return samples


print("RSNA 샘플 빌드 중...")
rsna_samples = build_rsna_samples()
print(f"RSNA 샘플 완료: {len(rsna_samples)}장")


class XrayDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data    = data
        self.augment = augment

        self.base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = load_rsna_image(item['dcm_path'])

        if item['label'] == 1.0 and item['bboxes']:
            mask = make_rsna_mask(item['bboxes'])
        else:
            mask = np.zeros((224, 224), dtype=np.float32)

        label = item['label']

        if self.augment:
            img = self.aug_transform(img)
        else:
            img = self.base_transform(img)

        mask  = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return img, mask, label


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    rsna_labels = [int(s['label']) for s in rsna_samples]
    train, temp, train_lbl, temp_lbl = train_test_split(
        rsna_samples, rsna_labels, test_size=0.2, stratify=rsna_labels, random_state=42)
    val, test, _, _ = train_test_split(
        temp, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

    train_ds = XrayDataset(train, augment=True)
    val_ds   = XrayDataset(val,   augment=False)
    test_ds  = XrayDataset(test,  augment=False)

    print(f"\nTrain: {len(train_ds)}장")
    print(f"Val:   {len(val_ds)}장")
    print(f"Test:  {len(test_ds)}장")

    img, mask, label = train_ds[0]
    print(f"이미지 shape: {img.shape}")
    print(f"마스크 shape: {mask.shape}")
    print(f"라벨: {label}")
    print("\nRSNA Dataset 완료!")