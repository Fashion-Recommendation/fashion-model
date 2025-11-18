# src/train_mlp.py

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

PAIRS_PATH = "data/processed/train_pairs.csv"
EMB_PATH = "data/processed/item_embeddings_text.pkl"
MODEL_PATH = "models/mlp_text_mlp_v1.pkl"


def load_embeddings(path):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    print(f"[INFO] Loaded embeddings: {len(emb)} items")
    return emb


def load_pairs(path):
    pairs = pd.read_csv(path)
    print(f"[INFO] Loaded pairs: {len(pairs)} rows")
    return pairs


def build_features(pairs, emb_dict, sample_n=None):
    # 임베딩 있는 pair만 사용
    mask = pairs["item_id_1"].isin(emb_dict.keys()) & pairs["item_id_2"].isin(
        emb_dict.keys()
    )
    pairs = pairs[mask].reset_index(drop=True)
    print(f"[INFO] After filtering by embeddings: {len(pairs)} rows")

    # 개발용으로 일부만 쓰고 싶으면 sample_n 지정 (없으면 전체 사용)
    if sample_n is not None and sample_n < len(pairs):
        pairs = pairs.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled pairs: {len(pairs)} rows")

    dim = len(next(iter(emb_dict.values())))
    X = np.zeros((len(pairs), dim * 2), dtype=np.float32)
    y = pairs["label"].values.astype(np.int64)

    print("[INFO] Building feature matrix (concat emb1, emb2)...")
    for i, row in tqdm(enumerate(pairs.itertuples(index=False)), total=len(pairs)):
        e1 = emb_dict[row.item_id_1]
        e2 = emb_dict[row.item_id_2]
        X[i] = np.concatenate([e1, e2])

    return X, y


def train_mlp(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Train size:", X_train.shape, "Val size:", X_val.shape)

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        batch_size=1024,
        max_iter=10,
        verbose=True,
        random_state=42,
    )

    print("[INFO] Training MLP...")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_val, y_pred))

    return clf


def main(sample_n=None):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    emb_dict = load_embeddings(EMB_PATH)
    pairs = load_pairs(PAIRS_PATH)

    X, y = build_features(pairs, emb_dict, sample_n=sample_n)

    clf = train_mlp(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"\n[INFO] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    # sample_n=None이면 전체 사용.
    # 메모리/속도 걱정되면 예: sample_n=200000 이런 식으로 먼저 돌려봐도 됨.
    main(sample_n=None)
