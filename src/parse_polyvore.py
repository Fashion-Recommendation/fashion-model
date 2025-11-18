from pathlib import Path
import pandas as pd

# 프로젝트 루트 기준: data/polyvore_raw/train_no_dup.json
DATA_DIR = Path("data/polyvore_raw")

def load_outfits(split="train"):
    """
    train_no_dup.json / valid_no_dup.json / test_no_dup.json
    에서 outfit별로 set_id, item_ids 리스트만 뽑는 함수
    """
    json_name = {
        "train": "train_no_dup.json",
        "valid": "valid_no_dup.json",
        "test":  "test_no_dup.json",
    }[split]

    json_path = DATA_DIR / json_name
    print(f"loading {json_path} ...")

    df = pd.read_json(json_path)

    outfits = []
    for _, row in df.iterrows():
        set_id = row["set_id"]
        items = row["items"]  # 리스트

        item_ids = []
        for item in items:
            # Polyvore 관례: set_id_index 형태로 item id 만들기
            idx = item["index"]   # ex) 0, 1, 2 ...
            item_id = f"{set_id}_{idx}"
            item_ids.append(item_id)

        outfits.append({
            "set_id": set_id,
            "item_ids": item_ids,
        })

    return pd.DataFrame(outfits)

if __name__ == "__main__":
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        df_out = load_outfits(split)
        out_path = out_dir / f"{split}_outfits.csv"
        df_out.to_csv(out_path, index=False)
        print(f"saved: {out_path} ({len(df_out)} rows)")
