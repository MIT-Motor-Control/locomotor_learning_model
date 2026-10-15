"""Offline integrity, interrupted-transfer and safe-extraction checks."""

import hashlib
import http.client
import importlib.util
import io
import json
import stat
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_data.py"
spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def release(tmp_path, entries=None):
    contents = entries or [
        (f"{download.DATA_ROOT}/README.md", b"Dataset provenance\n"),
        (f"{download.DATA_ROOT}/processed_csv/example.csv", b"time,value\n0,1\n"),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in contents:
            archive.writestr(name, content)
    payload = buffer.getvalue()
    folder = tmp_path / "pieces"
    folder.mkdir()
    parts = []
    for number, start in enumerate(range(0, len(payload), 128), 1):
        chunk = payload[start : start + 128]
        name = f"{download.ARCHIVE_NAME}.part{number:03d}"
        (folder / name).write_bytes(chunk)
        parts.append({"name": name, "bytes": len(chunk), "sha256": digest(chunk)})
    manifest = {
        "schema_version": 1,
        "archive": {
            "name": download.ARCHIVE_NAME,
            "bytes": len(payload),
            "sha256": digest(payload),
        },
        "parts": parts,
    }
    (folder / download.MANIFEST_NAME).write_text(json.dumps(manifest))
    return manifest, folder, payload


class Response(io.BytesIO):
    def __init__(self, data, *, length=True):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))} if length else {}


def test_offline_install_preserves_local_pieces_and_unrelated_cache(tmp_path):
    manifest, parts, _ = release(tmp_path)
    originals = {p.name: p.read_bytes() for p in parts.iterdir()}
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "unrelated.txt").write_text("keep me")
    destination = tmp_path / "data" / download.DATA_ROOT
    result = download.install(manifest, destination, cache, parts)
    assert result["files"] == 2
    assert result["archive_sha256"] == manifest["archive"]["sha256"]
    assert (destination / "processed_csv/example.csv").read_bytes() == b"time,value\n0,1\n"
    assert {p.name: p.read_bytes() for p in parts.iterdir()} == originals
    assert list(cache.iterdir()) == [cache / "unrelated.txt"]


def test_cleanup_failure_reports_successful_install(tmp_path, monkeypatch, capsys):
    manifest, parts, _ = release(tmp_path)
    cache = tmp_path / "cache"
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == cache / download.ARCHIVE_NAME:
            raise PermissionError("test cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    destination = tmp_path / "data"
    result = download.install(manifest, destination, cache, parts)
    assert result["files"] == 2
    assert (destination / "README.md").exists()
    assert (cache / download.ARCHIVE_NAME).is_file()
    assert "DATA installed; could not remove cached" in capsys.readouterr().out


def test_download_reuses_completed_piece_after_interruption(tmp_path, monkeypatch):
    manifest, parts, _ = release(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    first = manifest["parts"][0]["name"]
    (cache / first).write_bytes((parts / first).read_bytes())
    calls = []

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise KeyboardInterrupt
            return super().read(9)

    def interrupted(url):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        return Interrupted((parts / name).read_bytes())

    monkeypatch.setattr(download, "request", interrupted)
    destination = tmp_path / "data"
    with pytest.raises(KeyboardInterrupt):
        download.install(manifest, destination, cache)
    assert not destination.exists()
    second = manifest["parts"][1]["name"]
    assert (cache / f"{second}.partial").stat().st_size == 9
    assert not (cache / second).exists()

    def successful(url):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        return Response((parts / name).read_bytes())

    monkeypatch.setattr(download, "request", successful)
    download.install(manifest, destination, cache)
    assert first not in calls
    assert calls.count(second) == 2
    assert (destination / "README.md").read_text() == "Dataset provenance\n"
    assert not cache.exists()


@pytest.mark.parametrize("explicit_error", [True, False])
def test_incomplete_http_body_is_retried(tmp_path, monkeypatch, explicit_error):
    manifest, parts, _ = release(tmp_path)
    item = manifest["parts"][0]
    payload = (parts / item["name"]).read_bytes()
    attempts = []

    class Truncated(Response):
        def read(self, size=-1):
            if self.tell():
                if explicit_error:
                    raise http.client.IncompleteRead(b"", len(payload) - 9)
                return b""
            return super().read(9)

    def request(_):
        attempts.append(1)
        return Truncated(payload) if len(attempts) == 1 else Response(payload)

    monkeypatch.setattr(download, "request", request)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    cache = tmp_path / "cache"
    cache.mkdir()
    assert download.download_part(item, cache).read_bytes() == payload
    assert len(attempts) == 2


def test_oversized_response_stops_before_writing_excess(tmp_path, monkeypatch):
    item = {"name": f"{download.ARCHIVE_NAME}.part001", "bytes": 3, "sha256": digest(b"abc")}
    monkeypatch.setattr(download, "request", lambda _: Response(b"abcd", length=False))
    with pytest.raises(ValueError, match="exceeds expected size"):
        download.download_part(item, tmp_path)
    assert not (tmp_path / item["name"]).exists()
    assert (tmp_path / f"{item['name']}.partial").stat().st_size <= 3


@pytest.mark.parametrize("which", ["piece", "archive"])
def test_checksum_failure_prevents_extraction(tmp_path, which):
    manifest, parts, _ = release(tmp_path)
    if which == "piece":
        path = parts / manifest["parts"][0]["name"]
        path.write_bytes(b"!" + path.read_bytes()[1:])
    else:
        manifest["archive"]["sha256"] = "0" * 64
    destination = tmp_path / "data"
    with pytest.raises(ValueError, match="checksum"):
        download.install(manifest, destination, tmp_path / "cache", parts)
    assert not destination.exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/absolute.txt",
        "unrelated/file.txt",
        f"{download.DATA_ROOT}/../escape.txt",
        f"{download.DATA_ROOT}/sub\\escape.txt",
        f"{download.DATA_ROOT}/C:escape.txt",
        f"{download.DATA_ROOT}/a//b.txt",
        f"{download.DATA_ROOT}/trailing. ",
    ],
)
def test_unsafe_late_zip_member_leaves_no_destination(tmp_path, name):
    manifest, parts, _ = release(
        tmp_path, [(f"{download.DATA_ROOT}/valid.txt", b"valid"), (name, b"unsafe")]
    )
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="ZIP"):
        download.install(manifest, destination, tmp_path / "cache", parts)
    assert not destination.exists()
    assert not list(tmp_path.glob("data-extract-*"))
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize(
    "paths",
    [
        ("a.txt", "a.txt"),
        ("a.txt", "A.txt"),
        ("folder/a", "Folder/b"),
        ("file", "file/child"),
        ("file/child", "file"),
    ],
)
def test_duplicate_and_prefix_collisions_are_rejected(tmp_path, paths):
    with pytest.warns(UserWarning) if paths[0] == paths[1] else nullcontext():
        manifest, parts, _ = release(
            tmp_path, [(f"{download.DATA_ROOT}/{name}", b"content") for name in paths]
        )
    with pytest.raises(ValueError, match="Duplicate|collisi|collid"):
        download.install(manifest, tmp_path / "installed", tmp_path / "cache", parts)


def test_zip_symlink_is_rejected(tmp_path):
    link = zipfile.ZipInfo(f"{download.DATA_ROOT}/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    manifest, parts, _ = release(tmp_path, [(link, b"../../outside")])
    with pytest.raises(ValueError, match="links"):
        download.install(manifest, tmp_path / "installed", tmp_path / "cache", parts)


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_destination_is_preserved_before_network(tmp_path, monkeypatch, symlink):
    destination = tmp_path / "existing"
    if symlink:
        destination.symlink_to(tmp_path / "missing")
    else:
        destination.mkdir()
        (destination / "user.txt").write_text("do not modify")
    monkeypatch.setattr(download, "request", lambda _: pytest.fail("Network must not be used"))
    with pytest.raises(SystemExit) as error:
        download.main(["--destination", str(destination)])
    assert error.value.code == 1
    assert (
        destination.is_symlink()
        if symlink
        else (destination / "user.txt").read_text() == "do not modify"
    )


def test_cache_symlink_is_preserved(tmp_path):
    manifest, parts, _ = release(tmp_path)
    target = tmp_path / "user.txt"
    target.write_text("preserve")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / download.ARCHIVE_NAME).symlink_to(target)
    with pytest.raises(ValueError, match="link"):
        download.install(manifest, tmp_path / "data", cache, parts)
    assert target.read_text() == "preserve"
    assert (cache / download.ARCHIVE_NAME).is_symlink()


def test_manifest_is_pinned_and_arbitrary_remote_urls_are_rejected(tmp_path, monkeypatch):
    manifest, parts, _ = release(tmp_path)
    path = parts / download.MANIFEST_NAME
    with pytest.raises(ValueError, match="Manifest checksum"):
        download.load_manifest(str(path))
    monkeypatch.setattr(download, "request", lambda _: pytest.fail("No arbitrary remote request"))
    with pytest.raises(ValueError, match="pinned Zenodo"):
        download.load_manifest("https://example.org/manifest.json")
    monkeypatch.setattr(download, "MANIFEST_SHA256", digest(path.read_bytes()))
    with pytest.raises(ValueError, match="pinned DATA release archive"):
        download.load_manifest(str(path))
    monkeypatch.setattr(download, "ARCHIVE_BYTES", manifest["archive"]["bytes"])
    monkeypatch.setattr(download, "ARCHIVE_SHA256", manifest["archive"]["sha256"])
    assert download.load_manifest(str(path)) == manifest


def test_offline_cli_uses_manifest_beside_parts(tmp_path, monkeypatch):
    manifest, parts, _ = release(tmp_path)
    monkeypatch.setattr(
        download, "MANIFEST_SHA256", digest((parts / download.MANIFEST_NAME).read_bytes())
    )
    monkeypatch.setattr(download, "ARCHIVE_BYTES", manifest["archive"]["bytes"])
    monkeypatch.setattr(download, "ARCHIVE_SHA256", manifest["archive"]["sha256"])
    monkeypatch.setattr(
        download, "request", lambda _: pytest.fail("Offline installation used network")
    )
    destination = tmp_path / "data"
    assert (
        download.main(
            [
                "--parts-dir",
                str(parts),
                "--destination",
                str(destination),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert (destination / "README.md").exists()
