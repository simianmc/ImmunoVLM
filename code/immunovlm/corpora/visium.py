from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VisiumScaleFactors:
    tissue_hires_scalef: float
    spot_diameter_fullres: float


@dataclass(frozen=True)
class VisiumSpot:
    barcode: str
    in_tissue: bool
    array_row: int
    array_column: int
    pixel_row: float
    pixel_column: float


def read_tissue_positions(path: Path) -> tuple[VisiumSpot, ...]:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 6:
        frame = pd.read_csv(path)
    if frame.shape[1] != 6:
        raise ValueError("tissue positions table must contain six columns")
    output: list[VisiumSpot] = []
    for row in frame.itertuples(index=False, name=None):
        output.append(
            VisiumSpot(
                barcode=str(row[0]),
                in_tissue=bool(int(row[1])),
                array_row=int(row[2]),
                array_column=int(row[3]),
                pixel_row=float(row[4]),
                pixel_column=float(row[5]),
            )
        )
    return tuple(output)


def extract_spot_patch(
    image: Image.Image,
    spot: VisiumSpot,
    diameter: float,
    output_size: int = 224,
) -> Image.Image:
    radius = diameter / 2.0
    box = (
        round(spot.pixel_column - radius),
        round(spot.pixel_row - radius),
        round(spot.pixel_column + radius),
        round(spot.pixel_row + radius),
    )
    return image.crop(box).resize((output_size, output_size), Image.Resampling.LANCZOS)


def hexagonal_coordinates(spots: tuple[VisiumSpot, ...]) -> FloatArray:
    coordinates = np.asarray(
        [[spot.array_column, spot.array_row * np.sqrt(3.0) / 2.0] for spot in spots],
        dtype=np.float64,
    )
    coordinates -= coordinates.min(axis=0, keepdims=True)
    return coordinates


def pixel_coordinates(spots: tuple[VisiumSpot, ...]) -> FloatArray:
    return np.asarray([[spot.pixel_column, spot.pixel_row] for spot in spots], dtype=np.float64)


def spot_adjacency(spots: tuple[VisiumSpot, ...], neighbors: int = 6) -> npt.NDArray[np.int64]:
    coordinates = hexagonal_coordinates(spots)
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(difference**2, axis=-1))
    np.fill_diagonal(distance, np.inf)
    count = min(neighbors, max(1, len(spots) - 1))
    return np.argpartition(distance, kth=count - 1, axis=1)[:, :count].astype(np.int64)


def aggregate_regions(
    expression: FloatArray, spots: tuple[VisiumSpot, ...], neighbors: int = 6
) -> FloatArray:
    adjacency = spot_adjacency(spots, neighbors)
    combined = np.concatenate([expression[:, None, :], expression[adjacency]], axis=1)
    return combined.mean(axis=1)
