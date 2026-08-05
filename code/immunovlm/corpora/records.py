from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpotRecord:
    sample_id: str
    patient_id: str
    section_id: str
    disease: str
    subtype: str
    image_path: Path
    expression_path: Path
    coordinate_x: float
    coordinate_y: float
    tissue_fraction: float
    platform: str
    native_spatial: bool


@dataclass(frozen=True)
class CohortSpec:
    name: str
    disease: str
    platform: str
    root: Path
    manifest: Path
    expression_layer: str = "counts"
    image_magnification: float = 20.0
    native_spatial: bool = True
    external_only: bool = False


SUBTYPE_DESCRIPTIONS: dict[str, str] = {
    "LN_I": "Class I minimal mesangial lupus nephritis with normal glomeruli by light microscopy",
    "LN_II": "Class II mesangial proliferative lupus nephritis with mesangial hypercellularity",
    "LN_III": "Class III focal lupus nephritis involving fewer than half of glomeruli",
    "LN_IV": "Class IV diffuse lupus nephritis with active crescents and immune deposits",
    "LN_V": "Class V membranous lupus nephritis with subepithelial immune deposits",
    "LN_VI": "Class VI advanced sclerosing lupus nephritis with global glomerulosclerosis",
    "IBD_CD": "Crohn disease with transmural inflammation and discontinuous tissue involvement",
    "IBD_UC": "Ulcerative colitis with continuous mucosal inflammation limited to the colon",
    "IBD_IC": (
        "Indeterminate colitis with overlapping Crohn disease and ulcerative colitis features"
    ),
    "RA_LM": "Lympho myeloid rheumatoid synovitis with organized ectopic lymphoid structures",
    "RA_DM": "Diffuse myeloid rheumatoid synovitis with dispersed mononuclear infiltrates",
    "RA_PI": "Pauci immune rheumatoid synovitis with sparse inflammatory cell infiltration",
}


def disease_for_subtype(subtype: str) -> str:
    prefix = subtype.split("_", maxsplit=1)[0]
    mapping = {
        "LN": "lupus_nephritis",
        "IBD": "inflammatory_bowel_disease",
        "RA": "rheumatoid_arthritis",
    }
    if prefix not in mapping:
        raise KeyError(subtype)
    return mapping[prefix]


def class_index_map(subtypes: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(sorted(set(subtypes)))}
