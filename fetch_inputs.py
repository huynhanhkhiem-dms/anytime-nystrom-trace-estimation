"""Fetch the public research inputs and check them against the experiment digests."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import requests


HERE = Path(__file__).resolve().parent


def main():
    sources = json.loads((HERE / "data_sources.json").read_bytes())
    for source in sources:
        output = HERE / source["path"]
        if output.is_file() and sha256(output.read_bytes()).hexdigest() == source["sha256"]:
            print(f"Verified {source['path']}")
            continue
        response = requests.get(source["url"], timeout=(30, 180))
        response.raise_for_status()
        payload = response.content
        if sha256(payload).hexdigest() != source.get("download_sha256", source["sha256"]):
            raise ValueError(f"Download checksum mismatch for {source['path']}")
        if "member" in source:
            with ZipFile(BytesIO(payload)) as archive:
                payload = archive.read(source["member"])
        if sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError(f"Input checksum mismatch for {source['path']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
        print(f"Downloaded and verified {source['path']}")


if __name__ == "__main__":
    main()
