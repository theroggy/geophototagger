from pathlib import Path

from scripts.remove_duplicate_images import remove_internal_duplicates


def test_remove_internal_duplicates_keeps_first_path(tmp_path: Path) -> None:
    kept_path = tmp_path / "image__a.jpg"
    duplicate_path = tmp_path / "image__b.jpg"
    unrelated_path = tmp_path / "other__a.jpg"
    for path in (duplicate_path, unrelated_path, kept_path):
        path.write_bytes(b"image")

    removed_count = remove_internal_duplicates(tmp_path)

    assert removed_count == 1
    assert kept_path.exists()
    assert not duplicate_path.exists()
    assert unrelated_path.exists()
