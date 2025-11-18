from pathlib import Path
import pandas as pd
import random
from itertools import combinations
from ast import literal_eval
from tqdm import tqdm

random.seed(42)

DATA_DIR = Path("data/processed")

def load_train_outfits():
    path = DATA_DIR / "train_outfits.csv"
    # item_ids가 문자열이므로 리스트로 복원
    df = pd.read_csv(path, converters={"item_ids": literal_eval})
    return df

def make_positive_pairs(df):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="making positive pairs"):
        item_ids = row["item_ids"]
        if len(item_ids) < 2:
            continue
        for a, b in combinations(item_ids, 2):
            rows.append((a, b, 1))  # label = 1
    return rows

def make_negative_pairs(df, num_neg_per_pos=1):
    # item -> set_id 매핑 만들어서
    # 서로 다른 outfit에서만 negative 샘플링
    item_to_set = {}
    all_items = []

    for _, row in df.iterrows():
        set_id = row["set_id"]
        for item_id in row["item_ids"]:
            item_to_set[item_id] = set_id
            all_items.append(item_id)

    all_items = list(set(all_items))  # 중복 제거

    # positive 개수만큼 negative 곱하기
    pos_pairs = make_positive_pairs(df)
    num_neg = num_neg_per_pos * len(pos_pairs)

    neg_rows = []
    pbar = tqdm(range(num_neg), desc="making negative pairs")
    for _ in pbar:
        while True:
            a, b = random.sample(all_items, 2)
            if item_to_set[a] != item_to_set[b]:
                neg_rows.append((a, b, 0))  # label = 0
                break

    return neg_rows

if __name__ == "__main__":
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = load_train_outfits()

    print("→ positive pairs 생성 중...")
    pos_pairs = make_positive_pairs(df_train)
    print(f"positive pairs: {len(pos_pairs)}")

    print("→ negative pairs 생성 중...")
    neg_pairs = make_negative_pairs(df_train, num_neg_per_pos=1)
    print(f"negative pairs: {len(neg_pairs)}")

    all_rows = pos_pairs + neg_pairs
    pair_df = pd.DataFrame(all_rows, columns=["item_id_1", "item_id_2", "label"])

    out_path = out_dir / "train_pairs.csv"
    pair_df.to_csv(out_path, index=False)
    print(f"saved: {out_path} ({len(pair_df)} rows)")
