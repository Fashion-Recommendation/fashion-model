import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

PAIR_PATH = "data/processed/disjoint_pairs_train_embedded_only.csv"

EMB_PATH = "data/processed/item_image_embeddings_sampled.pkl"
SAVE_MODEL = "models/mlp_image_disjoint.pkl"


def build_features(df, emb_dict):
    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
        a = str(row["item_id_1"]).strip()
        b = str(row["item_id_2"]).strip()

        if a not in emb_dict or b not in emb_dict:
            continue

        emb_a = emb_dict[a]
        emb_b = emb_dict[b]

        feat = np.concatenate([emb_a, emb_b])
        X.append(feat)
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    print("[INFO] Loading embeddings...")
    with open(EMB_PATH, "rb") as f:
        emb_dict = pickle.load(f)

    emb_dict = {str(k): v for k, v in emb_dict.items()}

    print("[INFO] Loading pairs...")
    df = pd.read_csv(PAIR_PATH)

    print("[INFO] Building feature matrix...")
    X, y = build_features(df, emb_dict)

    print("[INFO] X shape:", X.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training MLP...")
    clf = MLPClassifier(hidden_layer_sizes=(512, 256),
                        max_iter=20,
                        activation="relu",
                        solver="adam",
                        verbose=True)

    clf.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    pred = clf.predict(X_val)
    prob = clf.predict_proba(X_val)[:, 1]

    print("Accuracy:", accuracy_score(y_val, pred))
    print("F1:", f1_score(y_val, pred))
    print("AUC:", roc_auc_score(y_val, prob))

    print("Classification report:")
    print(classification_report(y_val, pred))

    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    with open(SAVE_MODEL, "wb") as f:
        pickle.dump(clf, f)

    print(f"▶ Saved model to {SAVE_MODEL}")
    X, y = build_features(df, emb_dict)
    print("[INFO] Valid pairs after filtering:", len(y))
    print("[INFO] X shape:", X.shape)

if __name__ == "__main__":
    main()
