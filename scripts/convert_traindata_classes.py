"""Create reusable class mappings and simplified training-data CSV output."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

SOURCE_PATH = Path(r"X:\Monitoring\ControlefotosJRC\traindata.csv")
MAPPING_PATH = Path(r"X:\Monitoring\ControlefotosJRC\class_mappings.csv")
OUTPUT_PATH = Path(
    r"X:\Monitoring\ControlefotosJRC\traindata_with_simplified_classes.csv"
)
MAPPING_FIELDS = ["hoofdteelt", "source_class", "simplified_nl", "simplified_en"]


def read_source_pairs(source_path: Path) -> list[tuple[str, str]]:
    """Read distinct numeric/class pairs from the source CSV.

    Args:
        source_path: CSV containing ``HOOFDTEELT`` and ``CLASSES`` columns.

    Returns:
        Sorted distinct ``(HOOFDTEELT, CLASSES)`` pairs.

    Raises:
        ValueError: If required columns are missing or values are empty.
    """
    with source_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required = {"HOOFDTEELT", "CLASSES"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Source CSV must contain {sorted(required)}")
        pairs = {
            ((row.get("HOOFDTEELT") or "").strip(), (row.get("CLASSES") or "").strip())
            for row in reader
        }
    if any(not code or not class_name for code, class_name in pairs):
        raise ValueError("Source CSV contains empty HOOFDTEELT or CLASSES values")
    return sorted(pairs, key=lambda pair: (int(pair[0]), pair[1]))


def _slug(value: str) -> str:
    """Create a stable lowercase ASCII token from Dutch source text."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = re.sub(r"\([^)]*\)|-\s*(industrie|vers)\b", "", normalized)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "unknown"


def simplify_pair(source_class: str) -> tuple[str, str]:
    """Return deterministic Dutch and English labels for one source pair."""
    if source_class == "Maïs":
        return "mais", "maize"
    if source_class.startswith("Braakliggend land"):
        return "braakliggend", "fallow"
    if source_class == "Grasland":
        return "grasland", "grassland"
    if source_class == "Grasklaver":
        return "grasklaver", "grass_clover"
    if source_class == "Meerjarige fruitteelten (appel)":
        return "appel", "apple"
    if source_class == "Meerjarige fruitteelten (peer)":
        return "peer", "pear"
    if source_class == "Aardappelen (niet-vroege)":
        return "aardappelen", "potatoes"
    if source_class == "Suikerbieten":
        return "suikerbieten", "sugar_beet"
    if source_class == "Wintertarwe":
        return "wintertarwe", "winter_wheat"
    if source_class == "Wintergerst":
        return "wintergerst", "winter_barley"
    if source_class == "Triticale":
        return "triticale", "triticale"
    if source_class == "Voederbieten":
        return "voederbieten", "fodder_beet"
    if source_class == "Voedererwten (niet voor menselijke consumptie)":
        return "voedererwten", "fodder_peas"
    if source_class == "Eenjarige luzerne":
        return "luzerne", "annual_alfalfa"
    if source_class in {"Brocolli - vers", "Broccoli - vers"}:
        return "broccoli", "broccoli"
    if source_class == "Witloof (voor de wortel) - vers":
        return "witloof", "chicory"
    if source_class == "Stallen en gebouwen":
        return "gebouwen", "buildings"
    if source_class == "Hoofdgebouwen":
        return "hoofdgebouwen", "main_buildings"
    if source_class == "Andere gebouwen":
        return "andere_gebouwen", "other_buildings"
    return _slug(source_class), _slug(source_class)


def write_mapping(mapping_path: Path, pairs: list[tuple[str, str]]) -> None:
    """Write a deterministic mapping CSV for all observed source pairs."""
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8", newline="") as mapping_file:
        writer = csv.DictWriter(mapping_file, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        for code, source_class in pairs:
            simplified_nl, simplified_en = simplify_pair(source_class)
            writer.writerow(
                {
                    "hoofdteelt": code,
                    "source_class": source_class,
                    "simplified_nl": simplified_nl,
                    "simplified_en": simplified_en,
                }
            )


def load_mapping(mapping_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Load and validate a reusable class mapping CSV."""
    with mapping_path.open(encoding="utf-8-sig", newline="") as mapping_file:
        reader = csv.DictReader(mapping_file)
        if reader.fieldnames != MAPPING_FIELDS:
            raise ValueError(f"Mapping CSV must contain {MAPPING_FIELDS}")
        mapping: dict[tuple[str, str], tuple[str, str]] = {}
        for row in reader:
            key = (
                (row["hoofdteelt"] or "").strip(),
                (row["source_class"] or "").strip(),
            )
            values = (
                (row["simplified_nl"] or "").strip(),
                (row["simplified_en"] or "").strip(),
            )
            if not all(key) or not all(values):
                raise ValueError("Mapping CSV contains an empty field")
            if key in mapping:
                raise ValueError(f"Duplicate mapping pair: {key}")
            mapping[key] = values
    return mapping


def convert_csv(
    source_path: Path,
    mapping_path: Path,
    output_path: Path,
) -> Counter[str]:
    """Append simplified Dutch and English labels to a source CSV.

    Args:
        source_path: Original training-data CSV.
        mapping_path: Reusable mapping CSV keyed by HOOFDTEELT and CLASSES.
        output_path: Destination CSV with appended simplified columns.

    Returns:
        Counts of rows by simplified Dutch label.

    Raises:
        ValueError: If a source pair is missing from the mapping.
    """
    mapping = load_mapping(mapping_path)
    with source_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        source_fields = reader.fieldnames or []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    *source_fields,
                    "CLASSES_SIMPLIFIED_NL",
                    "CLASSES_SIMPLIFIED_EN",
                ],
            )
            writer.writeheader()
            counts: Counter[str] = Counter()
            for row in reader:
                key = (row["HOOFDTEELT"].strip(), row["CLASSES"].strip())
                if key not in mapping:
                    raise ValueError(f"Unmapped source pair: {key}")
                simplified_nl, simplified_en = mapping[key]
                row["CLASSES_SIMPLIFIED_NL"] = simplified_nl
                row["CLASSES_SIMPLIFIED_EN"] = simplified_en
                writer.writerow(row)
                counts[simplified_nl] += 1
    return counts


def main() -> None:
    """Generate the mapping CSV and convert the source training-data CSV."""
    pairs = read_source_pairs(SOURCE_PATH)
    write_mapping(MAPPING_PATH, pairs)
    counts = convert_csv(SOURCE_PATH, MAPPING_PATH, OUTPUT_PATH)
    print(f"Converted {sum(counts.values())} rows")
    print(f"Wrote mapping: {MAPPING_PATH}")
    print(f"Wrote output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
