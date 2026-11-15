import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src.config import DATA_DIR, IMAGE_SIZE


# Standard ImageNet preprocessing (resize + center crop); normalisation is done later
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE),
])


def load_pet_dataset():
    # Oxford-IIIT Pet gives cat/dog images with pixel-level ground-truth masks
    return OxfordIIITPet(root=DATA_DIR, split="test", target_types="segmentation", download=True)


def get_sample(dataset, index):
    image, trimap = dataset[index]
    image = image_transform(image)

    # Trimap: 1 = pet, 2 = background, 3 = border -> pet + border is the object
    trimap = np.array(mask_transform(trimap))
    gt_mask = (trimap != 2).astype(np.uint8)
    return image, gt_mask


def normalize(image_batch):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image_batch - mean.to(image_batch.device)) / std.to(image_batch.device)
