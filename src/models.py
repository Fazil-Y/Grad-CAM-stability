import torch
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import normalize


def load_model(model_name, device):
    # Pretrained ImageNet models
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        target_layer = model.layer4[-1]
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        target_layer = model.features[29]   # last conv block output (after ReLU), as in the Grad-CAM paper
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # eval mode + no optimizer: weights are never updated (gradients are only needed for Grad-CAM)
    model = model.to(device).eval()
    return model, target_layer


def predict(model, image_batch, device):
    with torch.no_grad():
        logits = model(normalize(image_batch.to(device)))
    probs = logits.softmax(dim=1)
    return probs.argmax(dim=1).cpu(), probs.max(dim=1).values.cpu()


def compute_gradcam(cam, image_batch, target_class, device):
    # Vanilla Grad-CAM for the same target class on every image in the batch
    input_tensor = normalize(image_batch.to(device))
    targets = [ClassifierOutputTarget(target_class) for _ in range(len(image_batch))]
    return cam(input_tensor=input_tensor, targets=targets)   # (B, H, W), each map scaled to [0, 1]


def create_gradcam(model, target_layer):
    return GradCAM(model=model, target_layers=[target_layer])
