import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

from src import config
from src.dataset import get_sample
from src.models import load_model, predict, create_gradcam, compute_gradcam
from src.transformations import apply_transform

STABILITY_METRICS = ["spearman", "ssim", "iou_cams"]
METRIC_NAMES = {
    "spearman": "Spearman rank correlation",
    "ssim": "SSIM",
    "iou_cams": "IoU (original vs transformed CAM)",
    "iou_gt_trans": "IoU with ground-truth mask",
    "pointing_trans": "Pointing game accuracy",
}
TRANSFORM_ORDER = ["identity", "hflip", "translate", "brightness", "contrast", "noise", "rotate"]


def load_results(model_name):
    return pd.read_csv(os.path.join(config.RESULTS_DIR, f"{model_name}_results.csv"))


def save_figure(fig, name):
    os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(config.RESULTS_DIR, "figures", name), dpi=150, bbox_inches="tight")


def summary_table(results):
    # Mean and std across images for each transformation and strength (prediction-preserving cases only)
    kept = results[results["pred_same"] | results["transform"].isin(["random_pair", "center_baseline"])]
    table = kept.groupby(["transform", "strength"])[STABILITY_METRICS + ["iou_gt_trans", "pointing_trans"]]
    table = table.agg(["mean", "std"]).round(3)
    flip_rate = 1 - results.groupby(["transform", "strength"])["pred_same"].mean()
    table["pred_change_rate"] = flip_rate.round(3)
    return table


def plot_stability_bars(results, model_name):
    # Mean +- std of each stability metric per transformation type
    kept = results[results["pred_same"] | (results["transform"] == "random_pair")]
    order = TRANSFORM_ORDER + ["random_pair"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric in zip(axes, STABILITY_METRICS):
        grouped = kept.groupby("transform")[metric].agg(["mean", "std"]).reindex(order)
        ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=3, color="steelblue")
        ax.set_title(METRIC_NAMES[metric])
        ax.set_ylim(min(0, grouped["mean"].min() - 0.1), 1.05)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{model_name}: Grad-CAM stability per transformation (prediction unchanged)")
    fig.tight_layout()
    save_figure(fig, f"{model_name}_stability_bars.png")
    plt.show()


def plot_strength_curves(results, model_name):
    # Sensitivity analysis: metric vs transformation strength
    kept = results[results["pred_same"]]
    types = ["translate", "brightness", "contrast", "noise", "rotate"]
    fig, axes = plt.subplots(1, len(types), figsize=(20, 4))
    for ax, name in zip(axes, types):
        subset = kept[kept["transform"] == name]
        for metric in STABILITY_METRICS:
            grouped = subset.groupby("strength")[metric].agg(["mean", "std"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o", capsize=3,
                        label=METRIC_NAMES[metric])
        ax.set_title(name)
        ax.set_xlabel("strength")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{model_name}: stability vs transformation strength")
    fig.tight_layout()
    save_figure(fig, f"{model_name}_strength_curves.png")
    plt.show()


def plot_spearman_boxplot(results, model_name):
    # Spread across images for each transformation setting
    kept = results[results["pred_same"] & results["transform"].isin(TRANSFORM_ORDER)].copy()
    kept["setting"] = kept["transform"] + " " + kept["strength"].astype(str)
    settings = kept["setting"].unique()
    data = [kept.loc[kept["setting"] == s, "spearman"] for s in settings]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.boxplot(data, tick_labels=settings)
    ax.set_ylabel("Spearman rank correlation")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title(f"{model_name}: distribution of rank correlation across images")
    fig.tight_layout()
    save_figure(fig, f"{model_name}_spearman_boxplot.png")
    plt.show()


def plot_localization(results, model_name):
    # IoU with ground truth and pointing game: original image vs transformed vs centre baseline
    kept = results[results["pred_same"] & results["transform"].isin(TRANSFORM_ORDER)]
    original = results[results["transform"] == "identity"]
    center = results[results["transform"] == "center_baseline"]

    names = ["original"] + TRANSFORM_ORDER[1:] + ["center_baseline"]
    iou_means = [original["iou_gt_orig"].mean()]
    point_means = [original["pointing_orig"].mean()]
    for name in TRANSFORM_ORDER[1:]:
        iou_means.append(kept.loc[kept["transform"] == name, "iou_gt_trans"].mean())
        point_means.append(kept.loc[kept["transform"] == name, "pointing_trans"].mean())
    iou_means.append(center["iou_gt_trans"].mean())
    point_means.append(center["pointing_trans"].mean())

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, iou_means, width=0.4, label="IoU with GT mask")
    ax.bar(x + 0.2, point_means, width=0.4, label="Pointing game accuracy")
    ax.set_xticks(x, names, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"{model_name}: localisation of Grad-CAM vs ground-truth pet mask")
    fig.tight_layout()
    save_figure(fig, f"{model_name}_localization.png")
    plt.show()


def plot_model_comparison(all_results):
    # Compare architectures on mean Spearman per transformation type
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(all_results)
    x = np.arange(len(TRANSFORM_ORDER))
    for i, (model_name, results) in enumerate(all_results.items()):
        kept = results[results["pred_same"]]
        means = kept.groupby("transform")["spearman"].mean().reindex(TRANSFORM_ORDER)
        ax.bar(x + i * width, means, width=width, label=model_name)
    ax.set_xticks(x + width * (len(all_results) - 1) / 2, TRANSFORM_ORDER)
    ax.set_ylabel("Mean Spearman rank correlation")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Grad-CAM stability: ResNet-50 vs VGG-16")
    fig.tight_layout()
    save_figure(fig, "model_comparison.png")
    plt.show()


def show_heatmaps(dataset, model_name, image_index, settings=None):
    # Overlay Grad-CAM on the original and transformed images (all in their own coordinates)
    if settings is None:
        settings = [("identity", 0), ("hflip", 1), ("translate", 0.08), ("brightness", 1.2),
                    ("contrast", 0.8), ("noise", 0.05), ("rotate", 10)]
    model, target_layer = load_model(model_name, config.DEVICE)
    cam = create_gradcam(model, target_layer)

    image, _ = get_sample(dataset, image_index)
    target_class = int(predict(model, image[None], config.DEVICE)[0][0])
    generator = torch.Generator().manual_seed(config.SEED + image_index)
    batch = torch.stack([apply_transform(image, n, s, generator) for n, s in settings])
    preds, confs = predict(model, batch, config.DEVICE)
    cams = compute_gradcam(cam, batch, target_class, config.DEVICE)

    fig, axes = plt.subplots(1, len(settings), figsize=(3 * len(settings), 3.4))
    for i, (name, strength) in enumerate(settings):
        rgb = batch[i].permute(1, 2, 0).numpy()
        axes[i].imshow(show_cam_on_image(rgb, cams[i], use_rgb=True))
        axes[i].set_title(f"{name} {strength}\npred {int(preds[i])} ({float(confs[i]):.2f})", fontsize=9)
        axes[i].axis("off")
    fig.suptitle(f"{model_name}, image {image_index}, target class {target_class}")
    fig.tight_layout()
    save_figure(fig, f"{model_name}_heatmaps_{image_index}.png")
    plt.show()


def worst_cases(results, n=3):
    # Images where the heatmap changed the most although the prediction stayed the same
    kept = results[results["pred_same"] & results["transform"].isin(TRANSFORM_ORDER[1:])]
    return kept.nsmallest(n, "spearman")[["image_index", "transform", "strength", "spearman", "ssim", "iou_cams"]]
