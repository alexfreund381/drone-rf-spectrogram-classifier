"""Official dataset source record. Values were read from the Zenodo API on 2026-11-10."""

from __future__ import annotations

ZENODO_RECORD_URL = "https://zenodo.org/records/21428081"
ZENODO_API_URL = "https://zenodo.org/api/records/21428081"
DATASET_DOI = "https://doi.org/10.5281/zenodo.21428081"
ARCHIVE_NAME = "LocalisingDroneRFEmissionsInSpectrograms.tar"
ARCHIVE_URL = f"{ZENODO_RECORD_URL}/files/{ARCHIVE_NAME}?download=1"
README_URL = f"{ZENODO_RECORD_URL}/files/README.md?download=1"
AUTHORS_CODE_URL = "https://github.com/juliankraft/LocalisingDroneRFEmissionsInSpectrograms"
ARCHIVE_SIZE_BYTES = 101_077_473_280
ARCHIVE_MD5 = "0e9e7f813de8d3816e6293bf37c3a8c6"
LICENSE = "CC BY 4.0"
ACCESSED = "2026-11-10"
ARCHIVE_ROOT = "LocalisingDroneRFEmissionsInSpectrograms"

# Filename tag -> human-readable radio system (from the dataset README).
RADIO_SYSTEMS: dict[str, str] = {
    "target0": "DJI",
    "target1": "FutabaT14",
    "target2": "FutabaT7",
    "target3": "Graupner",
    "target4": "Turnigy",
    "target5": "Taranis",
    "labnoise": "WiFi/Bluetooth background",
}
# Filename tag -> YOLO class id expected inside the label file.
TAG_TO_CLASS_ID: dict[str, int] = {f"target{i}": i for i in range(6)}
BACKGROUND_TAG = "labnoise"

CLASS_NAMES = ("background_rf", "drone_rf")  # index = binary label


def source_info() -> dict[str, object]:
    """Return the provenance record as a plain dictionary."""
    return {
        "zenodo_record": ZENODO_RECORD_URL,
        "doi": DATASET_DOI,
        "archive_url": ARCHIVE_URL,
        "archive_size_bytes": ARCHIVE_SIZE_BYTES,
        "archive_md5_published": ARCHIVE_MD5,
        "readme_url": README_URL,
        "authors_code": AUTHORS_CODE_URL,
        "license": LICENSE,
        "accessed": ACCESSED,
        "radio_systems": RADIO_SYSTEMS,
    }
