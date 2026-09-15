"""Metrics and charts written to reports/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (servers, Docker, CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, p, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, p)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "base_loss_rate": round(float(y_true.mean()), 4),
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_k_selection(scores: pd.DataFrame, best_k: int, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(scores["k"], scores["inertia"], "o-", color="tab:blue")
    ax1.set_xlabel("Number of segments (k)")
    ax1.set_ylabel("Inertia (elbow)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["silhouette"], "s--", color="tab:orange")
    ax2.set_ylabel("Silhouette", color="tab:orange")
    ax1.axvline(best_k, color="grey", ls=":", label=f"chosen k = {best_k}")
    ax1.legend(loc="upper center")
    ax1.set_title("Choosing the number of borrower segments")
    _save(fig, path)


def plot_segments(profile: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(profile["Segment"], profile["Loss_Rate"], color="tab:red")
    axes[0].set_xlabel("Loss rate (train)")
    axes[0].invert_yaxis()
    axes[1].barh(profile["Segment"], profile["Total_Outstanding"] / 1e7, color="tab:blue")
    axes[1].set_xlabel("Total outstanding (INR crore)")
    axes[1].invert_yaxis()
    fig.suptitle("Borrower segments")
    _save(fig, path)


def plot_roc_pr(y, p, baseline_score, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y, p)
    bfpr, btpr, _ = roc_curve(y, baseline_score)
    axes[0].plot(fpr, tpr, label=f"Model (AUC {roc_auc_score(y, p):.3f})")
    axes[0].plot(bfpr, btpr, label=f"Days-past-due rule (AUC {roc_auc_score(y, baseline_score):.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y, p)
    bprec, brec, _ = precision_recall_curve(y, baseline_score)
    axes[1].plot(rec, prec, label=f"Model (AP {average_precision_score(y, p):.3f})")
    axes[1].plot(brec, bprec, label=f"Days-past-due rule (AP {average_precision_score(y, baseline_score):.3f})")
    axes[1].axhline(np.mean(y), color="k", ls=":", lw=1, label="Random")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="upper right")
    _save(fig, path)


def plot_calibration(y, p, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    ax.set(xlabel="Predicted loss probability", ylabel="Observed loss rate", title="Calibration (test set)")
    ax.legend()
    _save(fig, path)


def plot_confusion(cm: dict, path: Path) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Recover", "Loss"])
    ax.set_yticks([0, 1], ["Recovered", "Written off"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test)")
    _save(fig, path)


def plot_threshold_curve(curve: pd.DataFrame, best_t: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["net_value"] / 1e7)
    ax.axvline(best_t, color="tab:red", ls="--", label=f"best threshold = {best_t:.2f}")
    ax.set(xlabel="Loss-probability threshold for escalation", ylabel="Net recovery value (INR crore)",
           title="Choosing the threshold by business value (validation set)")
    ax.legend()
    _save(fig, path)


def plot_importance(imp: pd.DataFrame, path: Path) -> None:
    imp = imp.sort_values("importance_mean").tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color="tab:green")
    ax.set(xlabel="Drop in PR-AUC when shuffled", title="Permutation importance (test set)")
    _save(fig, path)
