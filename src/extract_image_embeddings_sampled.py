import os
import pickle
from tqdm import tqdm
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

IMG_DIR = "polyvore_outfits/images"
ITEM_LIST_PATH = "data/processed/sample_item_ids.pkl"
SAVE_PATH = "data/processed/item_image_embeddings_sampled.pkl"


def main():
    print("[INFO] Loading sampled item_ids...")
    with open(ITEM_LIST_PATH, "rb") as f:
        item_ids = pickle.load(f)

    print(f"[INFO] Items to embed: {len(item_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    model.to(device).eval()

    emb_dict = {}

    for item_id in tqdm(item_ids, desc="Extracting sampled embeddings"):
        img_path = os.path.join(IMG_DIR, f"{item_id}.jpg")

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy().flatten()

        emb_dict[item_id] = emb

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"▶ Saved embeddings: {SAVE_PATH}")
    print(f"[INFO] Embedded items: {len(emb_dict)}")


if __name__ == "__main__":
    main()
