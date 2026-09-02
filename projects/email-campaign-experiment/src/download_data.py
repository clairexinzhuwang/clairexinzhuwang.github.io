"""Download and integrity-check the public Hillstrom experiment dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
EXPECTED_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path, force: bool = False) -> Path:
    output = output.resolve()
    if output.exists() and not force:
        observed = sha256_file(output)
        if observed == EXPECTED_SHA256:
            print(f"Already present and verified: {output}")
            return output
        raise RuntimeError(
            f"Refusing to overwrite {output}: checksum is {observed}. "
            "Use --force only after inspecting the file."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATA_URL, headers={"User-Agent": "hillstrom-case-study/1.0"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            mode="wb", prefix="hillstrom-", suffix=".csv.part", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)

        observed = sha256_file(temporary_path)
        if observed != EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded file failed checksum verification: "
                f"expected {EXPECTED_SHA256}, observed {observed}"
            )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Downloaded and verified: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download(arguments.output, arguments.force)

