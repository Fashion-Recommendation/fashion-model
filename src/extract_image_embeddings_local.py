import os
import json
import pickle
from tqdm import tqdm
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

DATA_ROOT = "polyvore_outfits"
IMG_DIR = os.path.join(DATA_ROOT, "images")
SAVE_PATH = "data/processed/item_image_embeddings.pkl"

SAMPLE_N = 5000  # 먼저 5천개 임베딩 → 나중에 전체 돌리면 됨


def extract_image_embeddings(save_path, sample_n=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    model.to(device).eval()

    # images 폴더 내 .jpg 파일명 = item_id
    filenames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
    item_ids = [f.replace(".jpg", "") for f in filenames]

    if sample_n is not None:
        item_ids = item_ids[:sample_n]

    emb_dict = {}

    for item_id in tqdm(item_ids, desc="Extracting image embeddings"):
        img_path = os.path.join(IMG_DIR, f"{item_id}.jpg")

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy().flatten()

        emb_dict[item_id] = emb

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"▶ Saved image embeddings to: {save_path}")
    print(f"[INFO] Embedded items: {len(emb_dict)}")


if __name__ == "__main__":
    extract_image_embeddings(SAVE_PATH, sample_n=SAMPLE_N)
