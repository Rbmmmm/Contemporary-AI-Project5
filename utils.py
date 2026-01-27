import os
import json
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from tqdm import tqdm


# ============================================================
# Basic Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 1. Save scalar metrics (per epoch)
# ============================================================

def save_metrics(
    save_dir: str,
    metrics: Dict[str, List[float]],
    filename: str = "metrics.json"
):
    """
    Save training/validation metrics to a JSON file.
    """
    ensure_dir(save_dir)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f">>> Metrics saved to {path}")


# ============================================================
# 2. Plot training curves
# ============================================================

def plot_curve(
    values: List[float],
    title: str,
    ylabel: str,
    save_path: str,
    xlabel: str = "Epoch"
):
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f">>> Plot saved to {save_path}")


def plot_training_curves(
    save_dir: str,
    train_loss: List[float],
    val_acc: List[float],
    val_f1: List[float] = None
):
    ensure_dir(save_dir)

    plot_curve(
        train_loss,
        title="Training Loss",
        ylabel="Loss",
        save_path=os.path.join(save_dir, "train_loss.png"),
    )

    plot_curve(
        val_acc,
        title="Validation Accuracy",
        ylabel="Accuracy",
        save_path=os.path.join(save_dir, "val_accuracy.png"),
    )

    if val_f1 is not None:
        plot_curve(
            val_f1,
            title="Validation Macro-F1",
            ylabel="F1 Score",
            save_path=os.path.join(save_dir, "val_f1.png"),
        )


# ============================================================
# 3. Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_dir: str,
    filename: str = "confusion_matrix.png",
    normalize: bool = True
):
    """
    Plot and save confusion matrix.
    """
    ensure_dir(save_dir)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f">>> Confusion matrix saved to {save_path}")


# ============================================================
# 4. Validation Evaluation (Confusion Matrix + Bad Cases)
# ============================================================

@torch.no_grad()
def evaluate_on_validation(
    model,
    val_loader,
    device,
    class_names,
    save_dir,
):
    """
    Evaluate best model on validation set:
    - confusion matrix
    - bad cases
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    bad_cases = []

    print(">>> Evaluating on validation set")

    for batch in tqdm(val_loader, desc="Evaluating on validation set"):
        image_feats = batch["image_features"].to(device)
        text_feats = batch["text_features"].to(device)
        labels = batch["labels"].to(device)
        guids = batch["guids"]

        # ----------------------------
        # Model branch
        # ----------------------------
        if hasattr(model, "class_prototypes"):
            # EmotionCLIP
            out = model(image_feats, text_feats, labels=labels)
            preds = out["preds"]
        else:
            # Baseline
            logits = model(image_feats, text_feats)
            preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        # bad cases
        for guid, gt, pred in zip(
            guids, labels.cpu().tolist(), preds.cpu().tolist()
        ):
            if gt != pred:
                bad_cases.append(
                    {
                        "guid": guid,
                        "gt": class_names[gt],
                        "pred": class_names[pred],
                    }
                )

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ----------------------------
    # Confusion matrix
    # ----------------------------
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        save_dir=save_dir,
        filename="val_confusion_matrix.png",
        normalize=True,
    )

    # ----------------------------
    # Bad cases
    # ----------------------------
    bad_case_path = os.path.join(save_dir, "bad_cases_val.txt")
    with open(bad_case_path, "w") as f:
        for item in bad_cases:
            f.write(
                f"{item['guid']}, gt={item['gt']}, pred={item['pred']}\n"
            )

    print(f">>> Bad cases saved to {bad_case_path}")
