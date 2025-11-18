import os
import json
import csv
import pickle
import random
from itertools import combinations
from tqdm import tqdm

DATA_ROOT = "polyvore_outfits"
DISJOINT_TRAIN = os.path.join(DATA_ROOT, "disjoint/train.json")
EMB_PATH = "data/processed/item_image_embeddings_sampled.pkl"


SAVE_PATH = "data/processed/disjoint_pairs_train_embedded_only.csv"

NEG_PER_POS = 1  # pos:neg = 1:1

def load_embeddings(path):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    return set(emb.keys())  # item_id set


def load_outfits(path, valid_item_ids):
    with open(path, "r") as f:
        data = json.load(f)

    outfits = []
    for outfit in data:
        items = [it["item_id"] for it in outfit["items"]]
        # 임베딩 있는 item만 필터링
        filtered = [i for i in items if i in valid_item_ids]
        if len(filtered) >= 2:
            outfits.append(filtered)

    return outfits


def generate_pairs(outfits, valid_item_ids):
    rows = []
    all_items = list(valid_item_ids)

    for outfit_items in tqdm(outfits, desc="Generating filtered pairs"):
        # positive
        pos_pairs = list(combinations(outfit_items, 2))
        for a, b in pos_pairs:
            rows.append((a, b, 1))

        # negative
        for _ in range(len(pos_pairs) * NEG_PER_POS):
            a = random.choice(outfit_items)
            b = random.choice(all_items)
            if b in outfit_items:
                continue
            rows.append((a, b, 0))

    return rows


def main():
    print("[INFO] Loading embeddings...")
    valid_items = load_embeddings(EMB_PATH)
    print(f"[INFO] #Embeddings available: {len(valid_items)}")

    print("[INFO] Loading outfits...")
    outfits = load_outfits(DISJOINT_TRAIN, valid_items)
    print(f"[INFO] Outfits usable: {len(outfits)}")

    print("[INFO] Generating pairs...")
    pairs = generate_pairs(outfits, valid_items)
    print(f"[INFO] Total pairs: {len(pairs)}")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id_1", "item_id_2", "label"])
        writer.writerows(pairs)

    print(f"▶ Saved filtered pair dataset to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
