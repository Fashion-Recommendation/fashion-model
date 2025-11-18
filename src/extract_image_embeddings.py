import os
import json
import pickle
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import torch
from transformers import CLIPProcessor, CLIPModel

RAW_PATH = "data/polyvore_raw/train_no_dup.json"
SAVE_PATH = "data/processed/item_embeddings_image_live.pkl"

SAMPLE_N = 5000  # 샘플링 수
MAX_LIVE = 800   # 살아있는 이미지 최대 몇 개까지 모을지 (자유롭게 조절)


def build_item_image_map(json_path, sample_n=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    # 모든 item_id, url 수집
    all_items = []
    for outfit in data:
        set_id = outfit["set_id"]
        for idx, item in enumerate(outfit["items"]):
            item_id = f"{set_id}_{idx}"
            url = item.get("image", None)
            if url and isinstance(url, str):
                all_items.append((item_id, url))

    # 샘플링
    import random
    random.seed(42)
    sampled = random.sample(all_items, min(sample_n, len(all_items)))

    print(f"[INFO] Sampled {len(sampled)} candidate items")
    return sampled


def check_live_urls(sampled_items, limit=MAX_LIVE):
    live = []
    print("[INFO] Checking which URLs are alive...")
    for item_id, url in tqdm(sampled_items, desc="Checking URLs"):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                live.append((item_id, url))
            if len(live) >= limit:
                break
        except:
            continue
    print(f"[INFO] Found {len(live)} live image URLs")
    return live


def extract_image_embeddings(live_items, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    model.to(device)
    model.eval()

    emb_dict = {}

    for item_id, url in tqdm(live_items, desc="Extracting embeddings"):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content)).convert("RGB")
        except:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy().flatten()

        emb_dict[item_id] = emb

    with open(save_path, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"▶ Saved image embeddings to {save_path}")
    print(f"[INFO] Total embeddings: {len(emb_dict)}")


if __name__ == "__main__":
    # 1) 샘플링
    candidates = build_item_image_map(RAW_PATH, sample_n=SAMPLE_N)

    # 2) 살아있는 URL만 필터링
    live_items = check_live_urls(candidates, limit=MAX_LIVE)

    # 3) 임베딩 생성
    extract_image_embeddings(live_items, SAVE_PATH)

    print("Done! 🎉")
