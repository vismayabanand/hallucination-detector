import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import average_precision_score
from torch.utils.data import TensorDataset, DataLoader

from src.evaluate import classification_metrics, correlation_metrics
from src.mlp_model import HallucinationMLP


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "top1_retrieval_score",
    "retrieval_mean",
    "retrieval_max",
    "entailment_mean",
    "entailment_max",
    "contradiction_mean",
    "contradiction_max",
    "neutral_mean",
    "neutral_max",
    "selfcheck_mean_sim",
    "selfcheck_max_sim",
    "selfcheck_support_count",
    "selfcheck_support_ratio",
]


def load_split(split_name):
    df = pd.read_csv(f"data/features/{split_name}_phase2_features.csv")
    X = df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return df, X, y


def make_loader(X, y, batch_size=64, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, loader):
    model.eval()
    all_probs = []
    all_y = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_y.extend(yb.numpy().tolist())

    metrics = classification_metrics(all_y, all_probs)
    metrics.update(correlation_metrics(all_y, all_probs))
    return metrics, np.array(all_probs), np.array(all_y)


def main():
    print("Device:", DEVICE)

    train_df, X_train, y_train = load_split("train")
    val_df, X_val, y_val = load_split("val")
    test_df, X_test, y_test = load_split("test")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=128, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=128, shuffle=False)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0.0, 1.0]),
        y=y_train
    )
    pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32).to(DEVICE)

    model = HallucinationMLP(input_dim=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_ap = -1.0
    best_state = None
    patience = 5
    patience_ctr = 0
    num_epochs = 25

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_metrics, val_probs, val_true = evaluate_model(model, val_loader)
        val_ap = average_precision_score(val_true, val_probs)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={total_loss / max(len(train_loader),1):.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc_pr={val_metrics['auc_pr']:.4f}"
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    val_metrics, _, _ = evaluate_model(model, val_loader)
    test_metrics, test_probs, test_true = evaluate_model(model, test_loader)

    print("\nValidation metrics:")
    print(val_metrics)

    print("\nTest metrics:")
    print(test_metrics)

    results = pd.DataFrame([
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])
    results.to_csv("results/tables/phase2_mlp_results.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_cols": FEATURE_COLS,
        },
        "results/models/phase2_mlp.pt"
    )

    with open("results/tables/phase2_feature_columns.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print("\nSaved:")
    print("results/tables/phase2_mlp_results.csv")
    print("results/models/phase2_mlp.pt")


if __name__ == "__main__":
    main()