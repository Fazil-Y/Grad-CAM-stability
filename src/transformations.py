import torch
import torchvision.transforms.functional as TF


def apply_transform(image, name, strength, generator=None):
    # image: tensor (3, H, W) with values in [0, 1]
    if name == "identity":
        return image.clone()
    if name == "hflip":
        return TF.hflip(image)
    if name == "translate":
        shift = int(round(strength * image.shape[-1]))
        return TF.affine(image, angle=0, translate=[shift, shift], scale=1.0, shear=[0.0])
    if name == "brightness":
        return TF.adjust_brightness(image, strength)
    if name == "contrast":
        return TF.adjust_contrast(image, strength)
    if name == "noise":
        noise = torch.randn(image.shape, generator=generator) * strength
        return (image + noise).clamp(0, 1)
    if name == "rotate":
        return TF.rotate(image, angle=strength)
    raise ValueError(f"Unknown transform: {name}")


def align_to_original(cam, name, strength):
    # Map a heatmap of the transformed image back to the original image coordinates.
    # Photometric transforms (brightness, contrast, noise) do not move pixels.
    cam = torch.from_numpy(cam)[None]
    if name == "hflip":
        cam = TF.hflip(cam)
    elif name == "translate":
        shift = int(round(strength * cam.shape[-1]))
        cam = TF.affine(cam, angle=0, translate=[-shift, -shift], scale=1.0, shear=[0.0])
    elif name == "rotate":
        cam = TF.rotate(cam, angle=-strength)
    return cam[0].numpy()


def valid_region(name, strength, size):
    # Pixels that are still inside the image after transforming and aligning back
    ones = torch.ones(1, size, size)
    moved = apply_transform(ones, name, strength) if name in ["translate", "rotate"] else ones
    back = align_to_original(moved[0].numpy(), name, strength)
    return back > 0.5
