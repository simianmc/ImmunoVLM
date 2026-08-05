from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision.transforms import functional as VF

ByteImage = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class MacenkoReference:
    stain_matrix: npt.NDArray[np.float64]
    maximum_concentration: npt.NDArray[np.float64]


def optical_density(image: ByteImage) -> npt.NDArray[np.float64]:
    pixels = image.astype(np.float64)
    return -np.log((pixels + 1.0) / 256.0)


def estimate_stain_matrix(image: ByteImage, threshold: float = 0.15) -> npt.NDArray[np.float64]:
    density = optical_density(image).reshape(-1, 3)
    density = density[np.all(density > threshold, axis=1)]
    covariance = np.cov(density, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    plane = vectors[:, 1:3]
    projected = density @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [1.0, 99.0])
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    stains = np.stack([first, second], axis=1)
    if stains[0, 0] < stains[0, 1]:
        stains = stains[:, ::-1]
    return stains


def fit_macenko_reference(image: ByteImage) -> MacenkoReference:
    stains = estimate_stain_matrix(image)
    density = optical_density(image).reshape(-1, 3).T
    concentration = np.linalg.lstsq(stains, density, rcond=None)[0]
    maximum = np.percentile(concentration, 99.0, axis=1)
    return MacenkoReference(stain_matrix=stains, maximum_concentration=maximum)


def macenko_normalize(image: ByteImage, reference: MacenkoReference) -> ByteImage:
    source_stains = estimate_stain_matrix(image)
    density = optical_density(image).reshape(-1, 3).T
    source_concentration = np.linalg.lstsq(source_stains, density, rcond=None)[0]
    source_maximum = np.percentile(source_concentration, 99.0, axis=1)
    scale = reference.maximum_concentration / np.maximum(source_maximum, 1e-8)
    concentration = source_concentration * scale[:, None]
    normalized = 255.0 * np.exp(-reference.stain_matrix @ concentration)
    return np.clip(normalized.T.reshape(image.shape), 0, 255).astype(np.uint8)


def tissue_fraction(image: ByteImage) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1]
    tissue = (gray < 230) & (saturation > 15)
    return float(tissue.mean())


class PatchTransform:
    def __init__(self, training: bool, size: int = 224) -> None:
        self.training = training
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = VF.resize(image, [self.size, self.size], antialias=True)
        if self.training:
            if bool(torch.rand(()) < 0.5):
                image = VF.hflip(image)
            if bool(torch.rand(()) < 0.5):
                image = VF.vflip(image)
            turns = int(torch.randint(0, 4, ()).item())
            image = VF.rotate(image, turns * 90)
            factors = 0.9 + 0.2 * torch.rand(3)
            image = VF.adjust_brightness(image, float(factors[0]))
            image = VF.adjust_contrast(image, float(factors[1]))
            image = VF.adjust_saturation(image, float(factors[2]))
        tensor = VF.pil_to_tensor(image).float() / 255.0
        return VF.normalize(tensor, self.mean, self.std)
