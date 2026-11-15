import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from src.config import CAM_THRESHOLD


def rank_correlation(cam_a, cam_b, valid):
    # Spearman rank correlation between two heatmaps (only pixels inside the image)
    corr = spearmanr(cam_a[valid], cam_b[valid]).statistic
    return 0.0 if np.isnan(corr) else float(corr)


def ssim(cam_a, cam_b, valid):
    a = np.where(valid, cam_a, 0)
    b = np.where(valid, cam_b, 0)
    return float(structural_similarity(a, b, data_range=1.0))


def iou(mask_a, mask_b):
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def cam_to_mask(cam, threshold=CAM_THRESHOLD):
    return cam >= threshold


def pointing_game(cam, gt_mask):
    # Hit if the maximum point of the heatmap falls on the object (used in the Grad-CAM paper)
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return int(gt_mask[y, x] > 0)


def renormalize(cam, valid):
    # Rescale an aligned heatmap to [0, 1] inside the valid region
    cam = np.where(valid, cam, 0)
    cam = cam - cam[valid].min()
    return cam / (cam[valid].max() + 1e-8)
