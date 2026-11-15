import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.dataset import load_pet_dataset, get_sample
from src.models import load_model, predict, create_gradcam, compute_gradcam
from src.transformations import apply_transform, align_to_original, valid_region
from src.metrics import rank_correlation, ssim, iou, cam_to_mask, pointing_game, renormalize


def is_pet_class(class_id):
    # ImageNet dog classes are 151-268 and cat classes are 281-285
    return 151 <= class_id <= 268 or 281 <= class_id <= 285


def center_heatmap(size):
    # Baseline: Gaussian blob in the image centre (no model involved)
    y, x = np.mgrid[0:size, 0:size]
    sigma = size / 4
    return np.exp(-((x - size / 2) ** 2 + (y - size / 2) ** 2) / (2 * sigma ** 2))


def get_transform_list():
    return [(name, s) for name, strengths in config.TRANSFORMS.items() for s in strengths]


def run_model(model_name, dataset):
    model, target_layer = load_model(model_name, config.DEVICE)
    cam = create_gradcam(model, target_layer)
    transform_list = get_transform_list()
    size = config.IMAGE_SIZE
    center_cam = center_heatmap(size)
    full_region = np.ones((size, size), dtype=bool)

    indices = list(range(len(dataset)))
    random.Random(config.SEED).shuffle(indices)

    rows = []
    previous_cam = None
    used = 0
    progress = tqdm(total=config.NUM_IMAGES, desc=model_name)

    for index in indices:
        if used == config.NUM_IMAGES:
            break
        image, gt_mask = get_sample(dataset, index)

        # Keep only images the model recognises as a dog or cat
        orig_pred, orig_conf = predict(model, image[None], config.DEVICE)
        orig_pred, orig_conf = int(orig_pred[0]), float(orig_conf[0])
        if not is_pet_class(orig_pred):
            continue

        # Build all transformed versions of the image
        generator = torch.Generator().manual_seed(config.SEED + index)
        batch = torch.stack([apply_transform(image, n, s, generator) for n, s in transform_list])
        preds, confs = predict(model, batch, config.DEVICE)

        orig_cam = compute_gradcam(cam, image[None], orig_pred, config.DEVICE)[0]
        trans_cams = compute_gradcam(cam, batch, orig_pred, config.DEVICE)

        for i, (name, strength) in enumerate(transform_list):
            valid = valid_region(name, strength, size)
            cam_a = renormalize(orig_cam, valid)
            cam_b = renormalize(align_to_original(trans_cams[i], name, strength), valid)
            gt = gt_mask * valid

            rows.append({
                "image_index": index,
                "transform": name,
                "strength": strength,
                "orig_class": orig_pred,
                "pred_class": int(preds[i]),
                "pred_same": int(preds[i]) == orig_pred,
                "orig_conf": orig_conf,
                "trans_conf": float(confs[i]),
                "spearman": rank_correlation(cam_a, cam_b, valid),
                "ssim": ssim(cam_a, cam_b, valid),
                "iou_cams": iou(cam_to_mask(cam_a) & valid, cam_to_mask(cam_b) & valid),
                "iou_gt_orig": iou(cam_to_mask(cam_a) & valid, gt),
                "iou_gt_trans": iou(cam_to_mask(cam_b) & valid, gt),
                "pointing_orig": pointing_game(cam_a, gt),
                "pointing_trans": pointing_game(cam_b, gt),
            })

        # Baseline 1: heatmap of a different image (what "unrelated" maps look like)
        if previous_cam is not None:
            rows.append({
                "image_index": index, "transform": "random_pair", "strength": 0,
                "orig_class": orig_pred, "pred_class": -1, "pred_same": False,
                "orig_conf": orig_conf, "trans_conf": np.nan,
                "spearman": rank_correlation(orig_cam, previous_cam, full_region),
                "ssim": ssim(orig_cam, previous_cam, full_region),
                "iou_cams": iou(cam_to_mask(orig_cam), cam_to_mask(previous_cam)),
                "iou_gt_orig": iou(cam_to_mask(orig_cam), gt_mask),
                "iou_gt_trans": iou(cam_to_mask(previous_cam), gt_mask),
                "pointing_orig": pointing_game(orig_cam, gt_mask),
                "pointing_trans": pointing_game(previous_cam, gt_mask),
            })

        # Baseline 2: centre Gaussian heatmap for localisation (IoU / pointing game)
        rows.append({
            "image_index": index, "transform": "center_baseline", "strength": 0,
            "orig_class": orig_pred, "pred_class": -1, "pred_same": False,
            "orig_conf": orig_conf, "trans_conf": np.nan,
            "spearman": rank_correlation(orig_cam, center_cam, full_region),
            "ssim": ssim(orig_cam, center_cam, full_region),
            "iou_cams": iou(cam_to_mask(orig_cam), cam_to_mask(center_cam)),
            "iou_gt_orig": iou(cam_to_mask(orig_cam), gt_mask),
            "iou_gt_trans": iou(cam_to_mask(center_cam), gt_mask),
            "pointing_orig": pointing_game(orig_cam, gt_mask),
            "pointing_trans": pointing_game(center_cam, gt_mask),
        })

        previous_cam = orig_cam
        used += 1
        progress.update(1)

    progress.close()
    del model, cam
    torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.insert(0, "model", model_name)
    return results


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    dataset = load_pet_dataset()

    for model_name in config.MODELS:
        results = run_model(model_name, dataset)
        path = os.path.join(config.RESULTS_DIR, f"{model_name}_results.csv")
        results.to_csv(path, index=False)
        print(f"Saved {len(results)} rows to {path}")


if __name__ == "__main__":
    main()
