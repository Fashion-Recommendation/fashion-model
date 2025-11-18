import pickle
import numpy as np
from tqdm import tqdm

EMB_PATH = "data/processed/item_image_embeddings_sampled.pkl"
MODEL_PATH = "models/mlp_image_disjoint.pkl"

def load_data():
    with open(EMB_PATH, "rb") as f:
        emb = pickle.load(f)
    emb = {str(k): v for k, v in emb.items()}

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return emb, model


def rank_items(anchor_id, top_k=5):
    emb_dict, model = load_data()

    anchor_id = str(anchor_id)

    if anchor_id not in emb_dict:
        raise ValueError(f"Item {anchor_id} not found in embeddings")

    anchor_emb = emb_dict[anchor_id]

    scores = []
    for item_id, emb in emb_dict.items():
        if item_id == anchor_id:
            continue

        feat = np.concatenate([anchor_emb, emb]).reshape(1, -1)
        score = model.predict_proba(feat)[0, 1]

        scores.append((item_id, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    # test
    test_id = "206214033"   # 존재하는 아이템 중 하나로 테스트
    print(rank_items(test_id, top_k=5))
