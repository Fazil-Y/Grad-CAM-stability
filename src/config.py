import os
import torch

# Paths (relative to the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Experiment settings
NUM_IMAGES = 300          # number of test images used (after filtering)
IMAGE_SIZE = 224
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models to compare (ResNet-50 = main model, VGG-16 = model used in the Grad-CAM paper)
MODELS = ["resnet50", "vgg16"]

# Threshold on the normalised Grad-CAM map to get a binary mask (for IoU)
CAM_THRESHOLD = 0.5

# Transformations and their strengths (from the project proposal)
TRANSFORMS = {
    "identity":    [0],
    "hflip":       [1],
    "translate":   [0.02, 0.05, 0.08],        # fraction of image size
    "brightness":  [0.8, 0.9, 1.1, 1.2],      # brightness factor
    "contrast":    [0.8, 0.9, 1.1, 1.2],      # contrast factor
    "noise":       [0.01, 0.03, 0.05],        # std of Gaussian noise (pixel range 0-1)
    "rotate":      [-10, -5, 5, 10],          # degrees
}
