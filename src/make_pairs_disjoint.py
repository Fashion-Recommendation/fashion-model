import os
import json
import random
import csv
from itertools import combinations
from tqdm import tqdm

DATA_ROOT = "polyvore_outfits"
DISJOINT_TRAIN = os.path.join(DATA_ROOT, "disjoint/train.json")
SAVE_PATH = "data/processed/disjoint_pairs_train.csv"

# negative 샘플 개수 비율 (1:1)
NEG_MULTIPLIER = 1

# 샘플 수 제한 (dev 용)
MAX_OUTFITS = None   # None이면 전체 사용
MAX_NEG_PER_OUTFIT = 50   # outfit 하나당 최대 negative 몇 개 생성할지


def load_outfits(path):
    with open(path, "r") as f:
        data = json.load(f)

    # outfit 구조: {"set_id": "...", "items": [{"item_id": "..."}]}
    outfits = []
    for outfit in data:
        item_ids = [it["item_id"] for it in outfit["items"]]
        if len(item_ids) >= 2:
            outfits.append(item_ids)
    return outfits


def generate_pairs(outfits, save_path):
    print(f"[INFO] Loaded {len(outfits)} outfits")

    # item pool for negative sampling
    all_items = [item for outfit in outfits for item in outfit]
    all_items = list(set(all_items))
    print(f"[INFO] Unique items: {len(all_items)}")

    rows = []

    for outfit_items in tqdm(outfits, desc="Generating pairs"):
        # --- Positive pairs ---
        pos_pairs = list(combinations(outfit_items, 2))
        for a, b in pos_pairs:
            rows.append((a, b, 1))

        # --- Negative pairs ---
        neg_needed = min(len(pos_pairs) * NEG_MULTIPLIER, MAX_NEG_PER_OUTFIT)
        for _ in range(neg_needed):
            a = random.choice(outfit_items)
            b = random.choice(all_items)
            if b in outfit_items:
                continue
            rows.append((a, b, 0))

    print(f"[INFO] Total generated pairs: {len(rows)}")

    # save CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id_1", "item_id_2", "label"])
        writer.writerows(rows)

    print(f"▶ Saved pairs to: {save_path}")


if __name__ == "__main__":
    all_outfits = load_outfits(DISJOINT_TRAIN)

    # 개발용: 일부만
    if MAX_OUTFITS is not None:
        all_outfits = all_outfits[:MAX_OUTFITS]

    generate_pairs(all_outfits, SAVE_PATH)
