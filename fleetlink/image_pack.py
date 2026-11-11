"""Pack a run's hydra image store into one compressed archive and rebuild it elsewhere.

Moved here from ``hydra_image_store.pack`` (the store reader, writer and
``check_store`` stay in ``hydra_image_store``). CLI::

    python -m fleetlink.image_pack pack RUN_DIR OUT.tar[.zst] [--frames keyframes|all]
        [--compression none|lossless|fast] [--zstd-level N] [--no-masks] [--quiet]
    python -m fleetlink.image_pack unpack ARCHIVE DEST_RUN_DIR [--store-name image_store]
        [--overwrite] [--no-check] [--dsg dsg.json] [--quiet]

(``hydra-image-store check`` checks a store.)

What is packed: the *indexed* store, not the raw image logs. Hydra/Khronos log
keyframe InputData zips (EXR depth, ~1.2 MB each), track dirs and event logs;
``hydra-image-store tail`` / ``index`` turn them into ``<run>/image_store`` with
JPEG colour and 16-bit PNG depth, about 7x smaller than the zips (Penn window A:
146 MB store vs 1.15 GB zips), and with every anchor, binding and alias already
resolved, so the destination needs no re-index (and no hydra_image_store
indexer). ``pack`` indexes a run that has raw logs but no up-to-date store first
(``hydra_image_store.indexer.ensure_indexed``; skipped while a tailer follows
it). The console's map transfer excludes the raw log dirs from its rsync for
the same reason.

``pack_store`` (source machine) writes a tar stream (optionally zstd-wrapped)::

    manifest.json            what was packed and how (first member)
    frames/<frame_id>.hisf   one encoded frame per shipped frame (image_transfer.encode_frame)
    files/<rel path>         kept observation mask PNGs, verbatim
    store.sqlite             consistent snapshot of the index (last member)

The snapshot keeps every row (sensors, frames, observations, bindings, aliases); image
paths of frames that were not shipped are set to NULL (exactly what ``gc`` does), and
shipped frames point at the paths ``unpack_store`` writes (``image_transfer.decoded_paths``).
``unpack_store`` (destination) streams the archive into ``<dest>/image_store.partial``,
restores downscaled depth to full resolution so the sensor calibration stays valid,
runs :func:`check_store` and only then renames it to ``<dest>/image_store``.

Both ends stream, so the console can run::

    ssh robot python -m fleetlink.image_pack pack RUN - --compression fast \\
        | python -m fleetlink.image_pack unpack - DEST

or in python ``pack_store(run, proc.stdin)`` / ``unpack_store(proc.stdout, dest)``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from typing import BinaryIO, Callable

from hydra_image_store import check_store
from hydra_image_store.store import DB_NAME, ImageStore, _Db

try:
    from hydra_image_store.check import resolve_store_dir
except ImportError:  # hydra_image_store before check.py (pack.py still in it)
    from hydra_image_store.store import _resolve_db

    def resolve_store_dir(path: str | os.PathLike) -> pathlib.Path:
        """``<run>/image_store`` for a run directory, else the store directory itself."""
        p = pathlib.Path(path).expanduser()
        if (p / "image_store" / DB_NAME).exists():
            return (p / "image_store").absolute()
        return _resolve_db(p).parent


from . import image_transfer as transfer

MANIFEST = "manifest.json"
FORMAT = "hydra-image-store-pack"
FORMAT_VERSION = 1
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _snapshot(src_db: pathlib.Path, dst: pathlib.Path) -> None:
    import urllib.parse

    uri = "file:" + urllib.parse.quote(str(src_db)) + "?mode=ro"
    src = sqlite3.connect(uri, uri=True)
    try:
        out = sqlite3.connect(str(dst))
        with out:
            src.backup(out)
        out.execute("PRAGMA journal_mode = DELETE")
        out.close()
    finally:
        src.close()


class _TarOut:
    def __init__(self, out, zstd_level: int | None):
        self._close: list[Callable] = []
        if isinstance(out, (str, os.PathLike)) and str(out) == "-":
            raw = sys.stdout.buffer
        elif isinstance(out, (str, os.PathLike)):
            raw = open(out, "wb")
            self._close.append(raw.close)
        else:
            raw = out
        self._counter = raw = _CountingWriter(raw)  # bytes on the wire (after zstd)
        self._zw = None
        if zstd_level is not None:
            zstd = transfer._zstd()
            self._zw = zstd.ZstdCompressor(level=zstd_level, threads=-1).stream_writer(
                raw, closefd=False
            )
            raw = self._zw
        self.tar = tarfile.open(fileobj=raw, mode="w|", format=tarfile.PAX_FORMAT)

    def add(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(time.time())
        info.mode = 0o644
        self.tar.addfile(info, io.BytesIO(data))

    def close(self) -> int:
        self.tar.close()
        if self._zw is not None:
            self._zw.flush(transfer._zstd().FLUSH_FRAME)
            self._zw.close()
        self._counter.flush()
        for c in self._close:
            c()
        return self._counter.n


class _CountingWriter(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def writable(self):
        return True

    def write(self, b):
        self.raw.write(b)
        self.n += len(b)
        return len(b)

    def flush(self):
        self.raw.flush()


def _ensure_indexed(src) -> None:
    """Build / refresh ``<run>/image_store`` from the run's raw image logs."""
    try:
        from hydra_image_store.indexer import ensure_indexed
    except ImportError:  # hydra_image_store before the indexer: store written by Hydra
        return
    try:
        ensure_indexed(src)
    except (OSError, sqlite3.Error) as exc:  # e.g. read-only medium: pack what is there
        print(f"image_pack: could not index {src}: {exc}", file=sys.stderr)


def pack_store(
    src: str | os.PathLike,
    out: str | os.PathLike | BinaryIO,
    *,
    frames: str = "keyframes",
    compression: str = "fast",
    zstd_level: int | None = None,
    include_masks: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Pack the image store of ``src`` (run dir or store dir) into ``out``.

    ``out``: a path (``.zst`` / ``.tzst`` suffix turns on outer zstd), ``"-"`` for
    stdout, or a writable binary file object. ``frames="keyframes"`` ships keyframes
    plus the frames of kept object observations (object images); ``"all"`` ships every
    frame with images. ``compression``: see :mod:`fleetlink.image_transfer`.
    ``zstd_level`` wraps the tar in zstd (mainly shrinks the sqlite snapshot; images are
    already compressed); default 3 for ``.zst`` paths, else off. ``progress(done,
    total)`` is called after each frame. Returns stats.
    """
    if frames not in transfer.FRAME_SELECTIONS:
        raise ValueError(f"frames must be one of {transfer.FRAME_SELECTIONS}")
    if compression not in transfer.COMPRESSION_MODES:
        raise ValueError(f"compression must be one of {transfer.COMPRESSION_MODES}")
    _ensure_indexed(src)
    store_dir = resolve_store_dir(src)
    db_path = store_dir / DB_NAME
    if not db_path.exists():
        raise FileNotFoundError(f"no image store at {store_dir}")
    if (
        zstd_level is None
        and isinstance(out, (str, os.PathLike))
        and str(out).endswith((".zst", ".tzst"))
    ):
        zstd_level = 3
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="his_pack_") as tmp:
        snap = pathlib.Path(tmp) / DB_NAME
        _snapshot(db_path, snap)
        db = _Db(snap, readonly=False)
        db.root = store_dir  # rows of the snapshot, files of the source store
        store = ImageStore([db])
        selected = transfer.select_frames(store, frames, include_kept_observations=True)
        manifest = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "source": str(store_dir),
            "frames": frames,
            "compression": compression,
            "created": time.time(),
            "frame_count": len(selected),
            "info": store.info()[0],
        }
        manifest["info"]["path"] = str(store_dir)
        tar = _TarOut(out, zstd_level)
        stats = {
            "frames": 0,
            "frame_bytes": 0,
            "rgb_bytes": 0,
            "depth_bytes": 0,
            "masks": 0,
            "mask_bytes": 0,
            "missing_files": 0,
        }
        try:
            tar.add(MANIFEST, json.dumps(manifest, indent=1).encode())
            conn = db.conn
            conn.execute("BEGIN")
            shipped = set()
            for i, frame in enumerate(selected):
                blob = transfer.encode_frame(store, frame, compression)
                header = transfer.decode_header(blob)
                if not header["parts"]:
                    stats["missing_files"] += 1
                    continue
                rgb_rel, depth_rel = _paths_from_header(header)
                conn.execute(
                    "UPDATE frames SET rgb_path = ?, depth_path = ? WHERE frame_id = ?",
                    (rgb_rel, depth_rel, frame.frame_id),
                )
                shipped.add(frame.frame_id)
                tar.add(f"frames/{frame.frame_id}.hisf", blob)
                stats["frames"] += 1
                stats["frame_bytes"] += len(blob)
                for p in header["parts"]:
                    stats[f"{p['name']}_bytes"] += int(p["size"])
                if progress is not None:
                    progress(i + 1, len(selected))
            # rows whose images were not shipped lose their paths
            conn.execute("CREATE TEMP TABLE shipped (frame_id INTEGER PRIMARY KEY)")
            conn.executemany("INSERT INTO shipped VALUES (?)", [(f,) for f in shipped])
            conn.execute(
                "UPDATE frames SET rgb_path = NULL, depth_path = NULL "
                "WHERE frame_id NOT IN (SELECT frame_id FROM shipped)"
            )
            masks = conn.execute(
                "SELECT obs_id, mask_path FROM observations WHERE mask_path IS NOT NULL"
            ).fetchall()
            for obs_id, rel in masks:
                path = store_dir / rel if rel else None
                safe = transfer._safe_rel(rel)
                if include_masks and safe and path.exists():
                    data = path.read_bytes()
                    tar.add(f"files/{safe}", data)
                    stats["masks"] += 1
                    stats["mask_bytes"] += len(data)
                    if safe != rel:
                        conn.execute(
                            "UPDATE observations SET mask_path = ? WHERE obs_id = ?",
                            (safe, obs_id),
                        )
                else:
                    conn.execute(
                        "UPDATE observations SET mask_path = NULL WHERE obs_id = ?",
                        (obs_id,),
                    )
            for k, v in (
                ("transfer_source", str(store_dir)),
                ("transfer_frames", frames),
                ("transfer_compression", compression),
                ("transfer_depth_downscale", "2" if compression == "fast" else "1"),
                ("transfer_packed_at", f"{time.time():.3f}"),
            ):
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (k, v)
                )
            conn.execute("COMMIT")
            conn.execute("DROP TABLE shipped")
            conn.execute("VACUUM")
            db.conn.close()
            tar.add(DB_NAME, snap.read_bytes())
        finally:
            stats["archive_bytes"] = tar.close()
    stats["seconds"] = round(time.monotonic() - t0, 3)
    stats["selected"] = len(selected)
    return stats


def _paths_from_header(header: dict) -> tuple[str | None, str | None]:
    fake = {
        "meta": header["meta"],
        "rgb": None,
        "rgb_format": None,
        "depth": None,
        "depth_png": None,
    }
    for p in header["parts"]:
        if p["name"] == "rgb":
            fake["rgb"], fake["rgb_format"] = b"", p["codec"]
        elif p["name"] == "depth":
            if p["codec"] == "delta-zstd":
                fake["depth"] = True
            else:
                fake["depth_png"], fake["depth_ext"] = b"", p.get("ext") or ".png"
    return transfer.decoded_paths(fake)


class _Prefixed(io.RawIOBase):
    """A readable stream with some already-read bytes put back in front."""

    def __init__(self, prefix: bytes, raw):
        self.prefix, self.raw = prefix, raw

    def readable(self):
        return True

    def readinto(self, b):
        if self.prefix:
            n = min(len(b), len(self.prefix))
            b[:n] = self.prefix[:n]
            self.prefix = self.prefix[n:]
            return n
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def _safe_member(name: str) -> str:
    p = pathlib.PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"unsafe archive member {name!r}")
    return str(p)


def unpack_store(
    src: str | os.PathLike | BinaryIO,
    dest: str | os.PathLike,
    *,
    store_name: str = "image_store",
    overwrite: bool = False,
    check: bool = True,
    dsg: str | os.PathLike | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Rebuild a packed store as ``<dest>/<store_name>`` (``dest`` is the run directory).

    ``src`` is an archive path, ``"-"`` (stdin) or a readable binary stream; zstd is
    detected from the magic. The store is assembled in ``<store>.partial`` and renamed
    into place only after :func:`check_store` passes (``check=False`` skips it; ``dsg``,
    default ``<dest>/hydra/backend/dsg.json`` when present, adds the scene graph checks).
    Returns stats including the check report.
    """
    dest = pathlib.Path(dest).expanduser().absolute()
    final = dest / store_name
    if final.exists() and not overwrite:
        raise FileExistsError(f"{final} exists (pass overwrite=True / --overwrite)")
    staging = dest / f"{store_name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    close = []
    if isinstance(src, (str, os.PathLike)) and str(src) == "-":
        raw = sys.stdin.buffer
    elif isinstance(src, (str, os.PathLike)):
        raw = open(src, "rb")
        close.append(raw)
    else:
        raw = src
    t0 = time.monotonic()
    stats = {"frames": 0, "files": 0, "bytes_in": 0}
    manifest = None
    try:
        head = raw.read(4)
        stream = _Prefixed(head, raw)
        if head == _ZSTD_MAGIC:
            stream = (
                transfer._zstd()
                .ZstdDecompressor()
                .stream_reader(stream, read_across_frames=True)
            )
        stream = io.BufferedReader(stream, 1 << 20)
        got_db = False
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = _safe_member(member.name)
                data = tar.extractfile(member).read()
                stats["bytes_in"] += len(data)
                if name == MANIFEST:
                    manifest = json.loads(data)
                    if manifest.get("format") != FORMAT:
                        raise ValueError("not a hydra-image-store pack archive")
                    if int(manifest.get("version", 0)) > FORMAT_VERSION:
                        raise ValueError(
                            f"archive version {manifest['version']} is newer than {FORMAT_VERSION}"
                        )
                elif manifest is None:
                    raise ValueError(
                        f"not a hydra-image-store pack archive ({name!r} before {MANIFEST})"
                    )
                elif name.startswith("frames/") and name.endswith(".hisf"):
                    transfer.write_decoded_images(staging, transfer.decode_frame(data))
                    stats["frames"] += 1
                    if progress is not None and manifest:
                        progress(stats["frames"], int(manifest.get("frame_count", 0)))
                elif name.startswith("files/"):
                    rel = name[len("files/") :]
                    out = transfer._inside(staging, _safe_member(rel))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
                    stats["files"] += 1
                elif name == DB_NAME:
                    (staging / DB_NAME).write_bytes(data)
                    got_db = True
        if manifest is None or not got_db:
            raise ValueError("incomplete archive (missing manifest or store.sqlite)")
        # the snapshot was stored with journal_mode=DELETE; switch the copy back to WAL
        conn = sqlite3.connect(str(staging / DB_NAME))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('transfer_unpacked_at', ?)",
            (f"{time.time():.3f}",),
        )
        conn.commit()
        conn.close()
        stats["manifest"] = {
            k: manifest[k] for k in ("source", "frames", "compression", "frame_count")
        }
        if check:
            if dsg is None and (dest / "hydra/backend/dsg.json").exists():
                dsg = dest / "hydra/backend/dsg.json"
            report = check_store(staging, dsg=dsg)
            stats["check"] = report
            if not report["ok"]:
                raise RuntimeError(
                    "unpacked store failed checks: "
                    + ", ".join(k for k, v in report["checks"].items() if not v)
                    + f" (left in {staging})"
                )
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
    finally:
        for f in close:
            f.close()
    stats["store"] = str(final)
    stats["seconds"] = round(time.monotonic() - t0, 3)
    return stats


def _progress_printer(label: str):
    def progress(done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            print(f"{label} {done}/{total}", file=sys.stderr, flush=True)

    return progress


def _archive_errors() -> tuple[type[BaseException], ...]:
    """Exceptions a corrupt, truncated or foreign archive raises while unpacking."""
    errors: list[type[BaseException]] = [
        tarfile.TarError,
        EOFError,
        json.JSONDecodeError,
    ]
    try:
        import zstandard

        errors.append(zstandard.ZstdError)
    except ImportError:
        pass
    return tuple(errors)


def _cmd_pack(args) -> int:
    try:
        stats = pack_store(
            args.run_dir,
            args.out,
            frames=args.frames,
            compression=args.compression,
            zstd_level=args.zstd_level,
            include_masks=not args.no_masks,
            progress=None if args.quiet else _progress_printer("packed"),
        )
    except (FileNotFoundError, ValueError, transfer.MissingDependency) as exc:
        print(f"pack failed: {exc}", file=sys.stderr)
        return 1
    # stdout may carry the archive; report on stderr
    print(json.dumps(stats, indent=1), file=sys.stderr)
    return 0


def _cmd_unpack(args) -> int:
    try:
        stats = unpack_store(
            args.archive,
            args.dest,
            store_name=args.store_name,
            overwrite=args.overwrite,
            check=not args.no_check,
            dsg=args.dsg,
            progress=None if args.quiet else _progress_printer("unpacked"),
        )
    except _archive_errors() as exc:
        print(
            f"unpack failed: not a readable pack archive (corrupt or truncated?): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    except (
        RuntimeError,
        OSError,
        ValueError,
        sqlite3.Error,
        transfer.MissingDependency,
    ) as exc:
        print(f"unpack failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fleetlink.image_pack",
        description="Pack a hydra image store into one archive / rebuild it elsewhere.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "pack", help="pack a run's image store into one compressed archive"
    )
    s.add_argument(
        "run_dir", help="run directory (uses <run>/image_store) or store directory"
    )
    s.add_argument("out", help="archive path (.tar, .tar.zst) or - for stdout")
    s.add_argument(
        "--frames",
        choices=list(transfer.FRAME_SELECTIONS),
        default="keyframes",
        help="keyframes (+ kept object observation frames) or every frame with images",
    )
    s.add_argument(
        "--compression", choices=list(transfer.COMPRESSION_MODES), default="fast"
    )
    s.add_argument(
        "--zstd-level",
        type=int,
        default=None,
        help="wrap the tar in zstd (default 3 for .zst paths, else off)",
    )
    s.add_argument(
        "--no-masks", action="store_true", help="do not ship kept mask PNGs (RLE stays)"
    )
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=_cmd_pack)

    s = sub.add_parser(
        "unpack", help="rebuild a packed store as <dest>/image_store and check it"
    )
    s.add_argument("archive", help="archive path or - for stdin (zstd is detected)")
    s.add_argument("dest", help="destination run directory")
    s.add_argument("--store-name", default="image_store")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--no-check", action="store_true")
    s.add_argument(
        "--dsg",
        help="scene graph for the checks (default <dest>/hydra/backend/dsg.json)",
    )
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=_cmd_unpack)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


__all__ = ["build_parser", "main", "pack_store", "unpack_store"]


if __name__ == "__main__":
    sys.exit(main())
