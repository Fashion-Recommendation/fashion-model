import os
import json
import random
import pickle

DATA_ROOT = "polyvore_outfits"
TRAIN_PATH = os.path.join(DATA_ROOT, "disjoint/train.json")

SAVE_ITEM_LIST = "data/processed/sample_item_ids.pkl"

# outfit 2000개 샘플링 (원하면 조절 가능)
N_OUTFITS = 2000


def load_train_outfits(path):
    with open(path, "r") as f:
        data = json.load(f)

    # outfit 리스트: [{"set_id": "...", "items": [{"item_id": "..."}]}]
    outfits = []
    for outfit in data:
        items = [it["item_id"] for it in outfit["items"]]
        if len(items) >= 2:
            outfits.append(items)
    return outfits


def sample_item_ids(outfits, n_outfits=2000):
    random.shuffle(outfits)
    selected = outfits[:n_outfits]

    # 중복 제거
    item_set = set()
    for outfit in selected:
        for item_id in outfit:
            item_set.add(item_id)

    print(f"[INFO] Outfits selected: {len(selected)}")
    print(f"[INFO] Unique item_ids: {len(item_set)}")

    return list(item_set)


def save_item_ids(item_ids, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(item_ids, f)
    print(f"▶ Saved sampled item_ids → {path}")


def main():
    print("[INFO] Loading train outfits...")
    outfits = load_train_outfits(TRAIN_PATH)

    print("[INFO] Sampling outfits & extracting item_ids...")
    item_ids = sample_item_ids(outfits, N_OUTFITS)

    save_item_ids(item_ids, SAVE_ITEM_LIST)


if __name__ == "__main__":
    main()
