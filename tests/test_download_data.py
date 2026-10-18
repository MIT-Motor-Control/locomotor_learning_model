"""Offline integrity, interrupted-transfer and safe-extraction checks."""

import hashlib
import http.client
import importlib.util
import io
import json
import stat
import urllib.error
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
    def __init__(self, data, *, length=True, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))} if length else {}
        self.headers.update(headers or {})


def pin_release(monkeypatch, manifest, parts):
    monkeypatch.setattr(download, "ARCHIVE_BYTES", manifest["archive"]["bytes"])
    monkeypatch.setattr(download, "ARCHIVE_SHA256", manifest["archive"]["sha256"])
    monkeypatch.setattr(
        download, "MANIFEST_SHA256", digest((parts / download.MANIFEST_NAME).read_bytes())
    )


def test_default_command_prefers_complete_zip(tmp_path, monkeypatch):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    calls = []

    def request(url, *, headers=None):
        calls.append(url)
        assert headers == {"Accept-Encoding": "identity"}
        assert url == download.RECORD_URL + download.ARCHIVE_NAME
        return Response(payload)

    monkeypatch.setattr(download, "request", request)
    destination = tmp_path / "data"
    assert (
        download.main(["--destination", str(destination), "--cache-dir", str(tmp_path / "cache")])
        == 0
    )
    assert len(calls) == 1
    assert (destination / "README.md").read_text() == "Dataset provenance\n"


def test_whole_zip_404_does_not_select_older_split_release(tmp_path, monkeypatch):
    manifest, parts, _ = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    calls = []

    def request(url, *, headers=None):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        if name == download.ARCHIVE_NAME:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        pytest.fail("Current release must not request the historical split files")

    monkeypatch.setattr(download, "request", request)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    with pytest.raises(FileNotFoundError, match="unpublished|access-restricted"):
        download.install_release(tmp_path / "data", tmp_path / "cache")
    assert calls == [download.ARCHIVE_NAME]
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("error_code", [403, 429, 500, None])
def test_network_errors_never_select_split_route(tmp_path, monkeypatch, error_code):
    calls = []

    def request(url, *, headers=None):
        calls.append(url)
        assert url.endswith(download.ARCHIVE_NAME)
        if error_code is None:
            raise TimeoutError("timeout")
        raise urllib.error.HTTPError(url, error_code, "test error", {}, None)

    monkeypatch.setattr(download, "request", request)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    with pytest.raises(OSError):
        download.install_release(tmp_path / "data", tmp_path / "cache")
    assert len(calls) == (1 if error_code == 403 else 3)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("error_code", [403, 404])
def test_unavailable_public_release_has_actionable_cli_error(tmp_path, monkeypatch, capsys, error_code):
    calls = []

    def request(url, **kwargs):
        calls.append(url)
        raise urllib.error.HTTPError(url, error_code, "unavailable", {}, None)

    monkeypatch.setattr(download, "request", request)
    monkeypatch.setattr(download.time, "sleep", lambda _: pytest.fail("Unavailable files must not retry"))
    with pytest.raises(SystemExit) as error:
        download.main(["--destination", str(tmp_path / "data"), "--cache-dir", str(tmp_path / "cache")])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert f"HTTP {error_code}" in output.err
    assert "unpublished or access-restricted" in output.err
    assert "--archive" in output.err
    assert "Connection interrupted" not in output.out
    assert len(calls) == 1
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("archive_in_cache", [False, True])
def test_local_archive_cli_verifies_installs_and_preserves_source(tmp_path, monkeypatch, archive_in_cache):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    cache = tmp_path / "cache"
    if archive_in_cache:
        cache.mkdir()
    archive = (cache if archive_in_cache else tmp_path) / "my DATA download.zip"
    archive.write_bytes(payload)
    before = archive.stat()
    destination = tmp_path / "installed"
    monkeypatch.setattr(download, "request", lambda *a, **k: pytest.fail("Local install used network"))
    assert download.main([
        "--archive", str(archive), "--destination", str(destination), "--cache-dir", str(cache)
    ]) == 0
    assert (destination / "README.md").read_text() == "Dataset provenance\n"
    assert archive.read_bytes() == payload
    assert archive.stat().st_mtime_ns == before.st_mtime_ns
    assert archive.stat().st_ino == before.st_ino
    assert cache.is_dir() if archive_in_cache else not cache.exists()


@pytest.mark.parametrize("corruption", ["size", "checksum"])
def test_local_archive_corruption_never_extracts_or_changes_source(tmp_path, monkeypatch, corruption):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    archive = tmp_path / "data.zip"
    invalid = payload[:-1] if corruption == "size" else b"!" + payload[1:]
    archive.write_bytes(invalid)
    before = archive.stat().st_mtime_ns
    monkeypatch.setattr(download, "extract", lambda *a: pytest.fail("Invalid ZIP reached extraction"))
    with pytest.raises(ValueError, match="size/checksum"):
        download.install_local_archive(archive, tmp_path / "installed")
    assert archive.read_bytes() == invalid
    assert archive.stat().st_mtime_ns == before
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_local_archive_requires_existing_regular_file(tmp_path, kind):
    archive = tmp_path / "data.zip"
    if kind == "directory":
        archive.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"preserve")
        archive.symlink_to(target)
    with pytest.raises((ValueError, FileNotFoundError)):
        download.install_local_archive(archive, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
    if kind == "symlink":
        assert archive.is_symlink()
        assert target.read_bytes() == b"preserve"


def test_local_archive_refuses_existing_destination_before_hashing(tmp_path, monkeypatch):
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "user.txt").write_text("preserve")
    monkeypatch.setattr(download, "sha256", lambda *a: pytest.fail("Existing destination must stop first"))
    with pytest.raises(FileExistsError, match="unchanged"):
        download.install_local_archive(tmp_path / "unused.zip", destination)
    assert (destination / "user.txt").read_text() == "preserve"


def test_local_archive_uses_safe_extraction(tmp_path, monkeypatch):
    manifest, parts, payload = release(tmp_path, [
        (f"{download.DATA_ROOT}/valid.txt", b"valid"), ("../escape.txt", b"unsafe")
    ])
    pin_release(monkeypatch, manifest, parts)
    archive = tmp_path / "data.zip"
    archive.write_bytes(payload)
    with pytest.raises(ValueError, match="ZIP"):
        download.install_local_archive(archive, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert archive.read_bytes() == payload


@pytest.mark.parametrize("legacy_flag", ["--parts-dir", "--manifest"])
def test_local_archive_cannot_mix_with_legacy_sources(tmp_path, monkeypatch, legacy_flag):
    monkeypatch.setattr(download, "install_local_archive", lambda *a: pytest.fail("Conflicting arguments reached installer"))
    with pytest.raises(SystemExit) as error:
        download.main(["--archive", str(tmp_path / "data.zip"), legacy_flag, "unused"])
    assert error.value.code == 2


@pytest.mark.parametrize("corruption", ["length", "checksum"])
def test_wrong_whole_zip_never_falls_back_or_extracts(tmp_path, monkeypatch, corruption):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    calls = []

    def request(url, *, headers=None):
        calls.append(url)
        assert url.endswith(download.ARCHIVE_NAME)
        invalid = payload[:-1] if corruption == "length" else b"!" + payload[1:]
        return Response(invalid)

    monkeypatch.setattr(download, "request", request)
    with pytest.raises(ValueError, match="size|checksum"):
        download.install_release(tmp_path / "data", tmp_path / "cache")
    assert len(calls) == 1
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("ignores_range", [False, True])
def test_whole_zip_resumes_or_safely_restarts_when_range_ignored(
    tmp_path, monkeypatch, ignores_range
):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{download.ARCHIVE_NAME}.partial"
    partial.write_bytes(payload[:43])

    def request(url, *, headers=None):
        assert headers == {"Accept-Encoding": "identity", "Range": "bytes=43-"}
        if ignores_range:
            return Response(payload)
        return Response(
            payload[43:],
            status=206,
            headers={"Content-Range": f"bytes 43-{len(payload) - 1}/{len(payload)}"},
        )

    monkeypatch.setattr(download, "request", request)
    result = download.install_release(tmp_path / "data", cache)
    assert result["files"] == 2
    assert (tmp_path / "data/README.md").read_text() == "Dataset provenance\n"
    assert not cache.exists()


@pytest.mark.parametrize("initial_offset", [0, 43])
def test_interrupted_whole_zip_retry_uses_new_offset(tmp_path, monkeypatch, initial_offset):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    calls = []
    cache = tmp_path / "cache"
    cache.mkdir()
    if initial_offset:
        (cache / f"{download.ARCHIVE_NAME}.partial").write_bytes(payload[:initial_offset])

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise http.client.IncompleteRead(b"", len(payload) - 17)
            return super().read(17)

    def request(url, *, headers=None):
        calls.append(headers)
        if len(calls) == 1:
            if initial_offset:
                assert headers["Range"] == f"bytes={initial_offset}-"
                return Interrupted(
                    payload[initial_offset:],
                    status=206,
                    headers={
                        "Content-Range": f"bytes {initial_offset}-{len(payload) - 1}/{len(payload)}"
                    },
                )
            return Interrupted(payload)
        offset = initial_offset + 17
        assert headers["Range"] == f"bytes={offset}-"
        return Response(
            payload[offset:],
            status=206,
            headers={"Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"},
        )

    monkeypatch.setattr(download, "request", request)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    result = download.install_release(tmp_path / "data", cache)
    assert result["files"] == 2
    assert len(calls) == 2


@pytest.mark.parametrize("invalid", ["start", "end", "total", "missing", "length", "encoding"])
def test_bad_range_response_preserves_partial_and_never_extracts(tmp_path, monkeypatch, invalid):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{download.ARCHIVE_NAME}.partial"
    partial.write_bytes(payload[:43])
    start, end, total = 43, len(payload) - 1, len(payload)
    start += invalid == "start"
    end -= invalid == "end"
    total += invalid == "total"
    headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
    if invalid == "missing":
        headers.clear()
    elif invalid == "length":
        headers["Content-Length"] = "1"
    elif invalid == "encoding":
        headers["Content-Encoding"] = "gzip"
    monkeypatch.setattr(
        download, "request", lambda *a, **k: Response(payload[43:], status=206, headers=headers)
    )
    with pytest.raises(ValueError, match="range|size|encoding"):
        download.install_release(tmp_path / "data", cache)
    assert partial.read_bytes() == payload[:43]
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("state", ["complete", "oversized", "corrupt_complete"])
def test_whole_partial_is_checked_before_request(tmp_path, monkeypatch, state):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{download.ARCHIVE_NAME}.partial"
    partial.write_bytes(
        payload + b"!"
        if state == "oversized"
        else b"!" + payload[1:]
        if state == "corrupt_complete"
        else payload
    )
    monkeypatch.setattr(download, "request", lambda *a, **k: pytest.fail("No network required"))
    if state == "complete":
        assert download.install_release(tmp_path / "data", cache)["files"] == 2
    else:
        with pytest.raises(ValueError, match="size|checksum"):
            download.install_release(tmp_path / "data", cache)
        assert not (tmp_path / "data").exists()


def test_corrupt_resumed_prefix_fails_whole_checksum(tmp_path, monkeypatch):
    manifest, parts, payload = release(tmp_path)
    pin_release(monkeypatch, manifest, parts)
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / f"{download.ARCHIVE_NAME}.partial"
    partial.write_bytes(b"!" + payload[1:43])
    monkeypatch.setattr(
        download,
        "request",
        lambda *a, **k: Response(
            payload[43:],
            status=206,
            headers={"Content-Range": f"bytes 43-{len(payload) - 1}/{len(payload)}"},
        ),
    )
    with pytest.raises(ValueError, match="checksum"):
        download.install_release(tmp_path / "data", cache)
    assert not partial.exists()
    assert not (tmp_path / "data").exists()


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
    assert download.download_file(item, cache).read_bytes() == payload
    assert len(attempts) == 2


def test_oversized_response_stops_before_writing_excess(tmp_path, monkeypatch):
    item = {"name": f"{download.ARCHIVE_NAME}.part001", "bytes": 3, "sha256": digest(b"abc")}
    monkeypatch.setattr(download, "request", lambda _: Response(b"abcd", length=False))
    with pytest.raises(ValueError, match="exceeds expected size"):
        download.download_file(item, tmp_path)
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


@pytest.mark.parametrize(
    "name",
    [
        f"{download.DATA_ROOT}/link",
        f"{download.DATA_ROOT}/.DS_Store",
        f"__MACOSX/{download.DATA_ROOT}/._README.md",
    ],
)
def test_zip_symlink_is_rejected(tmp_path, name):
    link = zipfile.ZipInfo(name)
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
    with pytest.raises(ValueError, match="pinned legacy DATA release archive"):
        download.load_manifest(str(path))
    monkeypatch.setattr(download, "LEGACY_ARCHIVE_BYTES", manifest["archive"]["bytes"])
    monkeypatch.setattr(download, "LEGACY_ARCHIVE_SHA256", manifest["archive"]["sha256"])
    assert download.ARCHIVE_SHA256 != manifest["archive"]["sha256"]
    assert download.load_manifest(str(path)) == manifest


def test_offline_cli_uses_manifest_beside_parts(tmp_path, monkeypatch):
    manifest, parts, _ = release(tmp_path)
    monkeypatch.setattr(
        download, "MANIFEST_SHA256", digest((parts / download.MANIFEST_NAME).read_bytes())
    )
    monkeypatch.setattr(download, "LEGACY_ARCHIVE_BYTES", manifest["archive"]["bytes"])
    monkeypatch.setattr(download, "LEGACY_ARCHIVE_SHA256", manifest["archive"]["sha256"])
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


def test_finder_metadata_is_validated_but_not_extracted(tmp_path):
    root = download.DATA_ROOT
    manifest, parts, _ = release(
        tmp_path,
        [
            (f"{root}/", b""),
            (f"{root}/processed_csv/example.csv", b"time,value\n0,1\n"),
            (f"{root}/README.md", b"Dataset provenance\n"),
            (f"{root}/.DS_Store", b"Finder view preferences"),
            ("__MACOSX/", b""),
            (f"__MACOSX/._{root}", b"directory attributes"),
            (f"__MACOSX/{root}/processed_csv/", b""),
            (f"__MACOSX/{root}/._processed_csv", b"directory attributes"),
            (f"__MACOSX/{root}/._README.md", b"file attributes"),
            (f"__MACOSX/{root}/processed_csv/._example.csv", b"file attributes"),
            (f"__MACOSX/{root}/._.DS_Store", b"Finder attributes"),
        ],
    )
    destination = tmp_path / "installed"
    result = download.install(manifest, destination, tmp_path / "cache", parts)
    assert result["files"] == 2
    assert sorted(str(p.relative_to(destination)) for p in destination.rglob("*")) == [
        "README.md", "processed_csv", "processed_csv/example.csv"
    ]
    assert (destination / "README.md").read_bytes() == b"Dataset provenance\n"
    assert not (tmp_path / "__MACOSX").exists()


@pytest.mark.parametrize(
    "name",
    [
        "__MACOSX/../escape.txt",
        f"__MACOSX/{download.DATA_ROOT}/../._escape.txt",
        f"__MACOSX/{download.DATA_ROOT}/sub\\._file.txt",
        f"__MACOSX/{download.DATA_ROOT}/._missing.txt",
        "__MACOSX/._unrelated",
        f"__MACOSX/{download.DATA_ROOT}/plain.txt",
        f"__MACOSX/{download.DATA_ROOT}/processed_csv/._..",
    ],
)
def test_unsafe_or_unmatched_finder_metadata_leaves_no_destination(tmp_path, name):
    manifest, parts, _ = release(
        tmp_path, [(f"{download.DATA_ROOT}/README.md", b"valid"), (name, b"invalid")]
    )
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="ZIP|metadata"):
        download.install(manifest, destination, tmp_path / "cache", parts)
    assert not destination.exists()
    assert not list(tmp_path.glob("data-extract-*"))


def test_finder_metadata_case_collision_is_rejected(tmp_path):
    root = download.DATA_ROOT
    manifest, parts, _ = release(
        tmp_path,
        [
            (f"{root}/folder/a", b"a"),
            (f"{root}/folder/b", b"b"),
            (f"__MACOSX/{root}/folder/._a", b"attributes"),
            (f"__MACOSX/{root}/Folder/._b", b"attributes"),
        ],
    )
    with pytest.raises(ValueError, match="Case-colliding"):
        download.install(manifest, tmp_path / "installed", tmp_path / "cache", parts)
