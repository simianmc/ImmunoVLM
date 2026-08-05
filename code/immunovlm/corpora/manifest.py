import csv
import hashlib
from pathlib import Path

from immunovlm.corpora.records import SpotRecord

REQUIRED_COLUMNS = {
    "sample_id",
    "patient_id",
    "section_id",
    "disease",
    "subtype",
    "image_path",
    "expression_path",
    "coordinate_x",
    "coordinate_y",
    "tissue_fraction",
    "platform",
    "native_spatial",
}


def read_manifest(path: Path, root: Path) -> list[SpotRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        records = [_parse_record(row, root) for row in reader]
    if not records:
        raise ValueError("manifest is empty")
    return records


def _parse_record(row: dict[str, str], root: Path) -> SpotRecord:
    return SpotRecord(
        sample_id=row["sample_id"],
        patient_id=row["patient_id"],
        section_id=row["section_id"],
        disease=row["disease"],
        subtype=row["subtype"],
        image_path=(root / row["image_path"]).resolve(),
        expression_path=(root / row["expression_path"]).resolve(),
        coordinate_x=float(row["coordinate_x"]),
        coordinate_y=float(row["coordinate_y"]),
        tissue_fraction=float(row["tissue_fraction"]),
        platform=row["platform"],
        native_spatial=row["native_spatial"].lower() in {"1", "true", "yes"},
    )


def validate_records(records: list[SpotRecord], minimum_tissue: float = 0.5) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            errors.append(f"duplicate sample_id {record.sample_id}")
        seen.add(record.sample_id)
        if not record.image_path.is_file():
            errors.append(f"missing image {record.image_path}")
        if not record.expression_path.is_file():
            errors.append(f"missing expression {record.expression_path}")
        if record.tissue_fraction < minimum_tissue:
            errors.append(f"insufficient tissue {record.sample_id}")
    return errors


def manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(records: list[SpotRecord], path: Path, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = sorted(REQUIRED_COLUMNS)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "patient_id": record.patient_id,
                    "section_id": record.section_id,
                    "disease": record.disease,
                    "subtype": record.subtype,
                    "image_path": record.image_path.relative_to(root),
                    "expression_path": record.expression_path.relative_to(root),
                    "coordinate_x": record.coordinate_x,
                    "coordinate_y": record.coordinate_y,
                    "tissue_fraction": record.tissue_fraction,
                    "platform": record.platform,
                    "native_spatial": record.native_spatial,
                }
            )
    temporary.replace(path)
