# src/extract_image_embeddings.py

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
SAVE_PATH = "data/processed/item_embeddings_image_5k.pkl"

# 🔥 5,000개만 뽑기
SAMPLE_N = 5000


def build_item_image_map(json_path, sample_n=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    image_items = []

    for outfit in data:
        set_id = outfit["set_id"]
        items = outfit["items"]

        for idx, item in enumerate(items):
            item_id = f"{set_id}_{idx}"
            image_url = item.get("image", {}).get("url", None)
            if image_url:
                image_items.append((item_id, image_url))

    # 🔥 여기서 샘플링
    if sample_n:
        import random
        random.seed(42)
        image_items = random.sample(image_items, sample_n)

    return dict(image_items)


def extract_image_embeddings(image_map, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    model.to(device)
    model.eval()

    emb_dict = {}

    for item_id, url in tqdm(image_map.items(), desc="Extracting image embeddings"):
        try:
            # 이미지 로드
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            continue

        # 인코딩
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            emb = outputs.cpu().numpy().flatten()

        emb_dict[item_id] = emb

    with open(save_path, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"▶ Saved image embeddings to: {save_path}")
    print(f"Total image embeddings: {len(emb_dict)}")


if __name__ == "__main__":
    print("Building image URL map...")
    image_map = build_item_image_map(RAW_PATH, sample_n=SAMPLE_N)

    print(f"Sample size: {len(image_map)} images")

    print("Extracting image embeddings...")
    extract_image_embeddings(image_map, SAVE_PATH)

    print("Done! 🎉")
