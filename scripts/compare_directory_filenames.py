"""Print filename keys present in both directories."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Print the sorted filename keys shared by each configured pair."""
    project = "maizemulch"
    trainingdata_dir = Path(f"X:/Monitoring/phototagger/{project}/trainingdata")
    testdata_dir = Path("Q:/phototagger/trainingdata_raw")

    directory_pairs = [
        (trainingdata_dir / "train", trainingdata_dir / "validation"),
        (testdata_dir / "varia/test", testdata_dir / "varia/test_extra"),
    ]
    print_duplicates = False
    unlink_duplicates_from_second_directory = True

    for directory1, directory2 in directory_pairs:
        # Determine duplicates based on filename keys
        print(f"Determine duplicates in: {directory1} <> {directory2}")
        files1 = filename_paths(directory1)
        files2 = filename_paths(directory2)
        duplicates = files1.keys() & files2.keys()

        # Print the number of duplicates for the current pair of directories
        print(f"Number of duplicates: {len(duplicates)}")
        if print_duplicates:
            for filename in sorted(duplicates):
                print(filename)

        if unlink_duplicates_from_second_directory:
            for filename in duplicates:
                for path in files2[filename]:
                    path.unlink()
                    print(f"Unlinked: {path}")


def filename_paths(directory: Path) -> dict[str, list[Path]]:
    """Index files by names truncated at the first double underscore."""
    files: dict[str, list[Path]] = {}
    for path in directory.iterdir():
        if path.is_file():
            filename_key = path.name.split("__", maxsplit=1)[0]
            files.setdefault(filename_key, []).append(path)
    return files


if __name__ == "__main__":
    main()
