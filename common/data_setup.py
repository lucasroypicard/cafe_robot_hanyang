"""Extract the public dataset once; keep notebook setup short and readable."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from tqdm.auto import tqdm


SCHEMA = "cafe_three_level_v10"
REQUIRED_FILES = ("index.json", "L1.json", "L2.json", "L3.json")
MARKER = ".extraction_complete.json"


def _check_index(index):
    if index.get("schema") != SCHEMA:
        raise ValueError("Use the current L1-L3 dataset.zip, not an older four-level archive.")
    if index.get("revision") != "l1_l2_l3_only_v1":
        raise ValueError("Use the updated L1-L3 dataset.zip and extract it into an empty DATA_DIR.")
    if set(index.get("allowed_splits", [])) != {"train", "val"}:
        raise ValueError("Select the student TRAIN/VAL dataset ZIP.")


def _check_folder(folder):
    for name in REQUIRED_FILES:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing data file: {folder / name}")
    _check_index(json.loads((folder / "index.json").read_text(encoding="utf-8")))
    if not (folder / "scenes").is_dir():
        raise FileNotFoundError(f"Missing scenes folder: {folder / 'scenes'}")


def prepare_dataset(archive_path, data_dir):
    """Unpack dataset.zip into DATA_DIR, or use a manually extracted folder.

    archive_path=None means the user already extracted the data. Otherwise a
    completion marker lets repeated notebook runs skip the same ZIP. Extraction
    uses a temporary folder so an interruption cannot look like a finished copy.
    Existing nonempty data is never overwritten.
    """
    destination = Path(data_dir).expanduser().resolve()
    if archive_path is None:
        _check_folder(destination)
        print("Using existing dataset:", destination)
        return destination

    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Check DATASET_ZIP: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if "dataset/index.json" in names:
            prefix = "dataset/"
        elif "index.json" in names:
            prefix = ""
        else:
            raise ValueError("Select dataset.zip containing dataset/index.json.")
        _check_index(json.loads(archive.read(prefix + "index.json")))
        if any(prefix + name not in names for name in REQUIRED_FILES):
            raise ValueError("The ZIP is missing level-wise request lists.")

        members = []
        seen = set()
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            if not entry.filename.startswith(prefix):
                raise ValueError("The ZIP contains files outside the dataset folder.")
            relative = PurePosixPath(entry.filename[len(prefix):])
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in entry.filename or not relative.parts
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("The ZIP contains an invalid file path.")
            if relative.as_posix() in seen or relative.as_posix() == MARKER:
                raise ValueError("The ZIP contains duplicate or reserved file paths.")
            seen.add(relative.as_posix())
            members.append((entry, relative))

        # ZIP records already contain each file's CRC and size. No second read
        # of a large Drive archive is needed to recognize an unchanged archive.
        contents = [(entry.filename, entry.CRC, entry.file_size) for entry, _ in members]
        fingerprint = hashlib.sha256(json.dumps(contents).encode()).hexdigest()
        marker = destination / MARKER
        if marker.is_file():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous.get("archive_fingerprint") == fingerprint:
                _check_folder(destination)
                print("Dataset already extracted:", destination)
                return destination
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(
                "DATA_DIR already contains files. For manually extracted data, use DATASET_ZIP=None. "
                "Extract a new ZIP into an empty DATA_DIR.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dataset_unpack_", dir=destination.parent) as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            for entry, relative in tqdm(members, desc="Extract dataset", unit="file"):
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                # ZipExtFile checks the CRC while reading; corrupt files raise.
                with archive.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            _check_folder(staging)
            (staging / MARKER).write_text(json.dumps({
                "archive_fingerprint": fingerprint, "files": len(members)
            }, indent=2) + "\n", encoding="utf-8")
            if destination.exists():
                destination.rmdir()  # Only the empty folder accepted above.
            staging.rename(destination)

    print("Dataset ready:", destination)
    return destination
