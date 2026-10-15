"""Download, verify and install the DATA release using Python's standard library.

Run from the repository root: python scripts/download_data.py
For downloaded pieces: python scripts/download_data.py --parts-dir /path/to/pieces
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

RECORD_URL = "https://zenodo.org/records/23402153/files/"
MANIFEST_NAME = "locomotor_learning_model-DATA.manifest.json"
DATA_ROOT = "locomotor_learning_model-DATA"
ARCHIVE_NAME = f"{DATA_ROOT}.zip"
MANIFEST_URL = RECORD_URL + MANIFEST_NAME
MANIFEST_SHA256 = "cace585e3d7ac1c869bf5df1ec10f6a25ba5e3e643cdfb611f5a864fee7bde5b"
ARCHIVE_SHA256 = "14e9e46710afffd3998d0f5a5f33933c42d5599f2856eb743e1f025f5ac17cea"
ARCHIVE_BYTES = 13_335_231_738
CHUNK_SIZE = 4 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def regular(path: Path) -> bool:
    """Reject links, including dangling links, before reading or replacing a file."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Expected a regular file, not a link or directory: {path}")
    return path.exists()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, item: dict) -> bool:
    return regular(path) and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]


def validate_manifest(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported DATA download manifest")
    archive, parts = value.get("archive"), value.get("parts")
    if not isinstance(archive, dict) or not isinstance(parts, list) or not parts:
        raise ValueError("Manifest must identify an archive and its ordered pieces")
    if archive.get("name") != ARCHIVE_NAME:
        raise ValueError("Manifest must describe the locomotor learning DATA archive")
    for item in [archive, *parts]:
        if not isinstance(item, dict) or type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("Every archive and piece needs a positive byte size")
        if not isinstance(item.get("sha256"), str) or not SHA256.fullmatch(item["sha256"]):
            raise ValueError("Every archive and piece needs a SHA-256 checksum")
    names = [f"{ARCHIVE_NAME}.part{number:03d}" for number in range(1, len(parts) + 1)]
    if [item.get("name") for item in parts] != names:
        raise ValueError("Manifest pieces must be uniquely named and ordered from part001")
    if sum(item["bytes"] for item in parts) != archive["bytes"]:
        raise ValueError("Manifest piece sizes do not add up to the archive size")
    return value


def request(url: str, *, headers: dict | None = None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "locomotor-learning-model/0.3.0", **(headers or {})}
        ),
        timeout=60,
    )


def load_manifest(location: str) -> dict:
    if location == MANIFEST_URL:
        with request(location) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("Download manifest exceeds the 1 MB size limit")
    else:
        if "://" in location:
            raise ValueError(
                "Only the pinned Zenodo release manifest URL or a local file is accepted"
            )
        path = Path(location)
        regular(path)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Download manifest exceeds the 1 MB size limit")
        payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Manifest checksum differs from the pinned DATA release")
    value = validate_manifest(json.loads(payload))
    if value["archive"]["sha256"] != ARCHIVE_SHA256 or value["archive"]["bytes"] != ARCHIVE_BYTES:
        raise ValueError("Manifest does not identify the pinned DATA release archive")
    return value


def free_space(directory: Path, needed: int):
    if shutil.disk_usage(directory).free < needed:
        raise OSError(f"Need at least {needed / 1e9:.2f} GB free for the next step in {directory}")


def download_file(
    item: dict, cache: Path, *, resume: bool = False, missing_ok: bool = False
) -> Path | None:
    target = cache / item["name"]
    if verified(target, item):
        print(f"Verified cached {item['name']}", flush=True)
        return target
    partial = target.with_name(target.name + ".partial")
    regular(partial)
    url = RECORD_URL + urllib.parse.quote(item["name"])
    for attempt in range(1, 4):
        offset = partial.stat().st_size if resume and partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError(
                f"Cached partial exceeds expected size; remove {partial} before retrying"
            )
        if offset == item["bytes"]:
            if verified(partial, item):
                os.replace(partial, target)
                return target
            partial.unlink()
            raise ValueError(f"Cached partial failed its checksum and was removed: {item['name']}")
        free_space(cache, item["bytes"] - offset)
        action = f"Resuming at {offset / 1e6:.1f} MB:" if offset else "Downloading"
        print(f"{action} {item['name']} ({item['bytes'] / 1e6:.1f} MB)", flush=True)
        digest, count = hashlib.sha256(), 0
        try:
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            response = request(url, headers=headers) if resume else request(url)
            with response:
                status = getattr(response, "status", 200)
                if status == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if not match or tuple(map(int, match.groups())) != (
                        offset,
                        item["bytes"] - 1,
                        item["bytes"],
                    ):
                        raise ValueError(f"Unexpected partial-download range for {item['name']}")
                elif status == 200:
                    # A server may ignore Range: restart rather than append the full response.
                    offset = 0
                else:
                    raise ValueError(f"Unexpected HTTP status {status} for {item['name']}")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError(f"Unexpected content encoding for {item['name']}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != item["bytes"] - offset:
                    raise ValueError(f"Unexpected download size for {item['name']}")
                if offset:
                    with partial.open("rb") as existing:
                        for block in iter(lambda: existing.read(CHUNK_SIZE), b""):
                            digest.update(block)
                    count = offset
                with partial.open("ab" if offset else "wb") as output:
                    while block := response.read(min(CHUNK_SIZE, item["bytes"] - count + 1)):
                        count += len(block)
                        if count > item["bytes"]:
                            raise ValueError(f"Download exceeds expected size: {item['name']}")
                        digest.update(block)
                        output.write(block)
            if count < item["bytes"]:
                raise http.client.IncompleteRead(b"", item["bytes"] - count)
            if digest.hexdigest() != item["sha256"]:
                if resume:
                    partial.unlink()
                raise ValueError(f"Downloaded file failed its size/checksum check: {item['name']}")
            os.replace(partial, target)
            return target
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            if missing_ok and isinstance(error, urllib.error.HTTPError) and error.code == 404:
                return None
            if attempt == 3:
                raise
            print(
                f"Connection interrupted; retrying this file (attempt {attempt + 1}/3)", flush=True
            )
            time.sleep(attempt)
    raise AssertionError("Unreachable download state")


def assemble(parts: list[Path], manifest: dict, cache: Path) -> Path:
    item = manifest["archive"]
    target = cache / item["name"]
    if verified(target, item):
        print("Verified previously assembled DATA archive", flush=True)
        return target
    partial = target.with_name(target.name + ".partial")
    regular(partial)
    free_space(cache, item["bytes"])
    digest, count = hashlib.sha256(), 0
    print("Assembling pieces and checking the complete archive", flush=True)
    with partial.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                    digest.update(block)
                    count += len(block)
                    output.write(block)
    if count != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise ValueError(
            "Complete DATA archive failed its size/checksum check; nothing was extracted"
        )
    os.replace(partial, target)
    return target


def zip_members(
    archive: zipfile.ZipFile,
) -> tuple[list[tuple[zipfile.ZipInfo, tuple[str, ...]]], int]:
    """Validate the entire ZIP before creating any extracted files."""
    result, nodes, spelling, explicit = [], {}, {}, set()
    total = 0
    for info in archive.infolist():
        name = info.orig_filename
        if "\\" in name or any(ord(c) < 32 for c in name) or any(c in name for c in ":<>|?*"):
            raise ValueError(f"Unsafe ZIP path: {name!r}")
        pieces = tuple(name.rstrip("/").split("/"))
        if (
            not pieces
            or pieces[0] != DATA_ROOT
            or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in pieces)
        ):
            raise ValueError(f"ZIP entry is outside the expected DATA folder: {name!r}")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or info.flag_bits & 1:
            raise ValueError(
                f"ZIP links, special files and encrypted files are unsupported: {name!r}"
            )
        is_dir = info.is_dir()
        if (mode == stat.S_IFDIR and not is_dir) or (len(pieces) == 1 and not is_dir):
            raise ValueError(f"ZIP entry has conflicting file/directory metadata: {name!r}")
        key = "/".join(pieces).casefold()
        if key in explicit:
            raise ValueError(f"Duplicate ZIP entry: {name!r}")
        explicit.add(key)
        for length in range(1, len(pieces) + 1):
            prefix = "/".join(pieces[:length])
            folded = prefix.casefold()
            if folded in spelling and spelling[folded] != prefix:
                raise ValueError(f"Case-colliding ZIP paths: {name!r}")
            spelling[folded] = prefix
            kind = "directory" if length < len(pieces) or is_dir else "file"
            if folded in nodes and nodes[folded] != kind:
                raise ValueError(f"ZIP file/directory collision: {name!r}")
            nodes[folded] = kind
        if not is_dir:
            total += info.file_size
            result.append((info, pieces))
    if not result:
        raise ValueError("DATA archive contains no regular files")
    return result, total


def extract(archive_path: Path, destination: Path) -> int:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists and will not be changed: {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        members, total = zip_members(archive)
        destination.parent.mkdir(parents=True, exist_ok=True)
        free_space(destination.parent, total)
        print(f"Extracting {len(members)} files ({total / 1e9:.2f} GB)", flush=True)
        with tempfile.TemporaryDirectory(prefix="data-extract-", dir=destination.parent) as scratch:
            staged = Path(scratch)
            for info, pieces in members:
                path = staged.joinpath(*pieces)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output, CHUNK_SIZE)
                if path.stat().st_size != info.file_size:
                    raise ValueError(f"Extracted file has the wrong size: {info.filename}")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Destination appeared during extraction: {destination}")
            os.rename(staged / DATA_ROOT, destination)
    return len(members)


def installation_paths(destination: Path, cache: Path, parts_dir: Path | None = None):
    destination, cache = Path(destination).absolute(), Path(cache).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Destination already exists and will not be changed: {destination}. "
            "Use the existing dataset with loco-publication prepare, or choose another --destination."
        )
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ValueError(f"Download cache must be a directory, not a link: {cache}")
    if cache.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Download cache must be outside the installation destination")
    if parts_dir is not None and cache.resolve() == Path(parts_dir).resolve():
        raise ValueError("Use a separate cache directory to preserve locally supplied pieces")
    cache.mkdir(parents=True, exist_ok=True)
    return destination, cache


def finish_install(
    archive: Path,
    item: dict,
    destination: Path,
    cache: Path,
    downloaded: list[Path],
    keep_downloads: bool,
) -> dict:
    count = extract(archive, destination)
    if not keep_downloads:
        for path in downloaded:
            try:
                path.unlink()
            except OSError as error:
                print(f"DATA installed; could not remove cached {path.name}: {error}", flush=True)
        # Never remove unrelated cache contents or locally supplied pieces.
        try:
            cache.rmdir()
        except OSError:
            pass
    return {"destination": str(destination), "files": count, "archive_sha256": item["sha256"]}


def install(
    manifest: dict,
    destination: Path,
    cache: Path,
    parts_dir: Path | None = None,
    *,
    keep_downloads: bool = False,
) -> dict:
    manifest = validate_manifest(manifest)
    destination, cache = installation_paths(destination, cache, parts_dir)
    parts = []
    for item in manifest["parts"]:
        if parts_dir is None:
            part = download_file(item, cache)
        else:
            part = Path(parts_dir) / item["name"]
            if not verified(part, item):
                raise ValueError(f"Local piece is missing or failed its checksum: {part}")
            print(f"Verified local {item['name']}", flush=True)
        parts.append(part)
    archive = assemble(parts, manifest, cache)
    return finish_install(
        archive,
        manifest["archive"],
        destination,
        cache,
        [archive, *(parts if parts_dir is None else [])],
        keep_downloads,
    )


def install_release(destination: Path, cache: Path, *, keep_downloads: bool = False) -> dict:
    """Prefer the complete pinned ZIP; only an HTTP404 selects the split route."""
    destination, cache = installation_paths(destination, cache)
    item = {"name": ARCHIVE_NAME, "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256}
    archive = download_file(item, cache, resume=True, missing_ok=True)
    if archive is not None:
        return finish_install(archive, item, destination, cache, [archive], keep_downloads)
    print("Complete DATA ZIP is absent; using the verified split release", flush=True)
    return install(load_manifest(MANIFEST_URL), destination, cache, keep_downloads=keep_downloads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", help="Use the split route with this local or pinned release manifest"
    )
    parser.add_argument(
        "--parts-dir", type=Path, help="Use pieces and manifest already downloaded here"
    )
    parser.add_argument("--destination", type=Path, default=Path("data") / DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/data-download/23402153"))
    parser.add_argument(
        "--keep-downloads", action="store_true", help="Retain verified cached ZIP and any pieces"
    )
    args = parser.parse_args(argv)
    try:
        # Refuse an existing dataset before requesting even the remote manifest.
        if args.destination.exists() or args.destination.is_symlink():
            raise FileExistsError(
                f"Destination already exists and is unchanged: {args.destination}. "
                "Run loco-publication prepare, or choose another --destination."
            )
        if args.parts_dir is not None or args.manifest is not None:
            location = args.manifest or str(args.parts_dir / MANIFEST_NAME)
            result = install(
                load_manifest(location),
                args.destination,
                args.cache_dir,
                args.parts_dir,
                keep_downloads=args.keep_downloads,
            )
        else:
            result = install_release(
                args.destination, args.cache_dir, keep_downloads=args.keep_downloads
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Cached downloads are retained; run the same command to continue.")
        return 130
    except (OSError, ValueError, zipfile.BadZipFile, http.client.HTTPException) as error:
        parser.exit(
            1, f"DATA installation stopped: {error}\nCached downloads are retained for retry.\n"
        )
    print(f"DATA installed in {result['destination']} ({result['files']} files).")
    print(f"Verified archive SHA-256: {result['archive_sha256']}")
    print("Next: loco-publication prepare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
