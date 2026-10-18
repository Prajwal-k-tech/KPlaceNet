#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract the Kerala slice of OSV-5M from HuggingFace into data/kerala_osv5m/.

Why: the deployed model fails badly on Kerala (green hills collapse into one
South-India cell; Kerala test photos miss by ~500-700 km). This script harvests
every Kerala image in OSV-5M (~16.9k train / ~0.36k test) for a Kerala
fine-tuning fallback and for a Kerala eval set. It never downloads the full
246 GB dataset: it reads remote zip central directories over HTTP range
requests and range-extracts only the Kerala members.

Phases (all resumable; caches under data/kerala_osv5m/cache):
  ids      download the coordinate index parquet, filter bbox -> ids_{split}.csv + town counts
  map      scan remote zip central directories -> zip_map.json (osv_id -> zip member)
  extract  range-download mapped members -> {split}/images/{osv_id}.jpg + metadata
  verify   merge metadata, validate images, apply gates -> extraction_report.json

Examples:
  python scripts/extract_kerala_osv5m.py ids
  python scripts/extract_kerala_osv5m.py map
  python scripts/extract_kerala_osv5m.py extract --limit 8 --dry-run
  python scripts/extract_kerala_osv5m.py all
"""

import argparse
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("[fatal] pandas is required (pip install pandas pyarrow)", flush=True)
    raise

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "kerala_osv5m"

DATA_REPO = "osv5m/osv5m"
REVISION = "cff33609b56b54d8743b7ee7a416eb8433e9a681"
INDEX_REPO = "giahuy4205/osv5m-geographic-index"

# Kerala bbox as (lat_min, lon_min, lat_max, lon_max); matches the lib research:
# 16,939 train rows / 357 test rows.
KERALA_BBOX = (8.0, 74.5, 12.8, 77.5)

# Sub-region boxes for the town-count breakdown, keyed by label.
TOWNS = {
    "kottayam": (9.35, 76.30, 9.85, 76.75),
    "munnar": (9.90, 76.90, 10.30, 77.25),
    "kochi": (9.75, 76.10, 10.15, 76.50),
    "trivandrum": (8.30, 76.75, 8.75, 77.20),
}

TRAIN_ID_GATE = (15000, 19000)  # expected ~16,939
MAP_COVERAGE_GATE = 0.95
EXTRACT_RATIO_GATE = 0.98

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def hf_url(rel_path):
    return f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{REVISION}/{rel_path}"


def san(rel_path):
    return rel_path.replace("/", "__")


def norm_id(value):
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def stem_key(s):
    """Normalize a member stem / id for loose matching (strip leading zeros)."""
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


def town_of(lat, lon):
    for name, (lat_min, lon_min, lat_max, lon_max) in TOWNS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "other_kerala"


# --------------------------------------------------------------------------
# RemoteZip: seekable read-only file-like over HTTP range requests
# --------------------------------------------------------------------------

class RemoteZip(io.RawIOBase):
    def __init__(self, url, retries=4, timeout=120):
        super().__init__()
        self.url = url
        self._resolved = None
        self._size = None
        self._pos = 0
        self._retries = retries
        self._timeout = timeout
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "kplacenet-kerala-extract/1.0"})
        return self._session

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def _ensure(self):
        if self._size is None:
            self._resolve()

    def _resolve(self):
        last = None
        for attempt in range(self._retries):
            try:
                r = self.session.get(
                    self.url,
                    headers={"Range": "bytes=0-0"},
                    timeout=self._timeout,
                    stream=True,
                )
                try:
                    if r.status_code in (200, 206):
                        cr = r.headers.get("Content-Range")
                        if cr and "/" in cr:
                            self._size = int(cr.rsplit("/", 1)[1])
                        else:
                            self._size = int(r.headers.get("Content-Length", "0")) or None
                        self._resolved = r.url
                        if self._size:
                            return
                    last = IOError(f"HTTP {r.status_code} resolving {self.url}")
                finally:
                    r.close()
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(2 ** attempt)
        raise IOError(f"cannot resolve remote size for {self.url}: {last}")

    def _read_range(self, start, end):
        headers = {"Range": f"bytes={start}-{end}"}
        last = None
        for attempt in range(self._retries):
            try:
                r = self.session.get(self._resolved, headers=headers, timeout=self._timeout)
                try:
                    if r.status_code == 403 and attempt == 0:
                        self._resolve()  # signed URL may have expired
                        last = IOError("HTTP 403, re-resolved")
                        continue
                    if r.status_code not in (200, 206):
                        last = IOError(f"HTTP {r.status_code} for bytes {start}-{end}")
                        time.sleep(2 ** attempt)
                        continue
                    data = r.content
                    if r.status_code == 200 and (end - start + 1) < len(data):
                        # server ignored Range; slice defensively
                        data = data[start : end + 1]
                    return data
                finally:
                    r.close()
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 ** attempt)
        raise IOError(f"range read failed {self._resolved} {start}-{end}: {last}")

    def read(self, n=-1):
        self._ensure()
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        out = bytearray()
        while len(out) < n and self._pos < self._size:
            want_end = min(self._pos + (n - len(out)) - 1, self._size - 1)
            chunk = self._read_range(self._pos, want_end)
            if not chunk:
                break
            out.extend(chunk)
            self._pos += len(chunk)
        return bytes(out)

    def seek(self, offset, whence=0):
        self._ensure()
        if whence == 0:
            pos = offset
        elif whence == 1:
            pos = self._pos + offset
        elif whence == 2:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(pos, self._size))
        return self._pos

    def tell(self):
        return self._pos


def open_remote_zip(url, retries=4):
    rz = RemoteZip(url, retries=retries)
    zf = zipfile.ZipFile(rz)
    return zf, rz


# --------------------------------------------------------------------------
# Phase: ids
# --------------------------------------------------------------------------

def phase_ids(args, splits):
    from huggingface_hub import hf_hub_download

    ids_dir = OUT_ROOT / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in splits:
        path = hf_hub_download(
            repo_id=INDEX_REPO,
            filename=f"data/{split}.parquet",
            repo_type="dataset",
            cache_dir=str(OUT_ROOT / "cache" / "hf"),
        )
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        log(f"[ids] {split}: parquet columns = {list(df.columns)} rows = {len(df)}")

        id_col = next((c for c in ("osv_id", "id", "image_id") if c in df.columns), None)
        lat_col = next((c for c in ("lat", "latitude") if c in df.columns), None)
        lon_col = next((c for c in ("lon", "lng", "longitude") if c in df.columns), None)
        if not (id_col and lat_col and lon_col):
            raise SystemExit(f"[fatal] unexpected index schema: {list(df.columns)}")

        lat_min, lon_min, lat_max, lon_max = KERALA_BBOX
        mask = (
            (df[lat_col] >= lat_min)
            & (df[lat_col] <= lat_max)
            & (df[lon_col] >= lon_min)
            & (df[lon_col] <= lon_max)
        )
        sub = df.loc[mask, [id_col, lat_col, lon_col]].rename(
            columns={id_col: "osv_id", lat_col: "lat", lon_col: "lon"}
        ).copy()
        sub["osv_id"] = sub["osv_id"].map(norm_id)
        sub["lat"] = sub["lat"].astype(float)
        sub["lon"] = sub["lon"].astype(float)
        sub["town"] = [town_of(a, b) for a, b in zip(sub["lat"], sub["lon"])]
        sub = sub.drop_duplicates(subset="osv_id").sort_values("osv_id").reset_index(drop=True)

        out_csv = ids_dir / f"ids_{split}.csv"
        sub.to_csv(out_csv, index=False)
        counts = sub["town"].value_counts().to_dict()
        results[split] = {"count": int(len(sub)), "towns": {k: int(v) for k, v in counts.items()}}
        log(f"[ids] {split}: {len(sub)} Kerala rows in bbox {KERALA_BBOX} -> {out_csv}")
        for name in sorted(counts, key=counts.get, reverse=True):
            log(f"[ids]   {name}: {counts[name]}")

    train_n = results.get("train", {}).get("count")
    if train_n is not None:
        lo, hi = TRAIN_ID_GATE
        verdict = "PASS" if lo <= train_n <= hi else "REVIEW"
        log(f"[ids] gate train rows in [{lo}, {hi}]: {train_n} -> {verdict}")
    return results


# --------------------------------------------------------------------------
# Phase: map
# --------------------------------------------------------------------------

def list_zips(split):
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=DATA_REPO, repo_type="dataset", revision=REVISION)
    prefix = f"images/{split}/"
    zips = sorted(f for f in files if f.startswith(prefix) and f.endswith(".zip"))
    return zips


def scan_zip_task(split, zrel, want_keys, members_dir, sizes_dir):
    """Scan one remote zip's central directory; cache member names + wanted sizes."""
    members_path = members_dir / f"{san(zrel)}.txt"
    sizes_path = sizes_dir / f"{san(zrel)}.csv"
    if members_path.exists() and members_path.stat().st_size > 0 and sizes_path.exists():
        return {"zip": zrel, "cached": True, "members": None, "matched": None}

    url = hf_url(zrel)
    last = None
    for attempt in range(3):
        try:
            zf, _rz = open_remote_zip(url)
            try:
                names = zf.namelist()
                matched = {}
                for name in names:
                    base = name.rsplit("/", 1)[-1]
                    stem = base.rsplit(".", 1)[0]
                    key = stem_key(stem)
                    if key in want_keys:
                        osv_id = want_keys[key]
                        try:
                            matched[osv_id] = zf.getinfo(name).file_size
                        except KeyError:
                            pass
            finally:
                zf.close()

            tmp = Path(str(members_path) + ".tmp")
            tmp.write_text("\n".join(names), encoding="utf-8")
            os.replace(tmp, members_path)
            tmp2 = Path(str(sizes_path) + ".tmp")
            with open(tmp2, "w", encoding="utf-8", newline="") as fh:
                fh.write("osv_id,size\n")
                for osv_id, size in matched.items():
                    fh.write(f"{osv_id},{size}\n")
            os.replace(tmp2, sizes_path)
            return {"zip": zrel, "cached": False, "members": len(names), "matched": len(matched)}
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    raise IOError(f"scan failed {zrel}: {last}")


def read_ids(split):
    path = OUT_ROOT / "ids" / f"ids_{split}.csv"
    if not path.exists():
        raise SystemExit(f"[fatal] missing {path}; run `ids` phase first")
    df = pd.read_csv(path, dtype={"osv_id": str})
    return df


def phase_map(args, splits):
    members_dir = OUT_ROOT / "cache" / "members"
    sizes_dir = OUT_ROOT / "cache" / "sizes"
    members_dir.mkdir(parents=True, exist_ok=True)
    sizes_dir.mkdir(parents=True, exist_ok=True)

    zip_map = {}
    summary = {}
    all_zips = {}
    for split in splits:
        zips = list_zips(split)
        all_zips[split] = zips
        (members_dir / split).mkdir(parents=True, exist_ok=True)
        sizes_dir_split = sizes_dir / split
        sizes_dir_split.mkdir(parents=True, exist_ok=True)

        ids_df = read_ids(split)
        want_keys = {stem_key(i): i for i in ids_df["osv_id"]}
        log(f"[map] {split}: {len(zips)} zips to scan, {len(want_keys)} wanted ids")

        todo = [
            z
            for z in zips
            if not (members_dir / split / f"{san(z)}.txt").exists()
        ]
        n_cached = len(zips) - len(todo)
        if n_cached:
            log(f"[map] {split}: {n_cached} zips already cached")
        done = n_cached
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(scan_zip_task, split, z, want_keys, members_dir / split, sizes_dir_split): z
                for z in todo
            }
            for fut in as_completed(futures):
                zrel = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"[map] FAILED scan {zrel}: {exc}")
                    continue
                done += 1
                if res["cached"]:
                    log(f"[map] {split}: {zrel} (cached) [{done}/{len(zips)}]")
                else:
                    log(
                        f"[map] {split}: {zrel}: {res['members']} members, "
                        f"{res['matched']} wanted [{done}/{len(zips)}]"
                    )

        # Build id -> (zip, member, size) mapping from caches.
        sizes_all = {}
        for z in zips:
            sizes_path = sizes_dir_split / f"{san(z)}.csv"
            if sizes_path.exists():
                with open(sizes_path, "r", encoding="utf-8") as fh:
                    next(fh, None)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        osv_id, size = line.rsplit(",", 1)
                        sizes_all[osv_id] = int(size)

        mapping = {}
        for z in zips:
            members_path = members_dir / split / f"{san(z)}.txt"
            if not members_path.exists():
                continue
            with open(members_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    name = line.strip()
                    if not name:
                        continue
                    base = name.rsplit("/", 1)[-1]
                    stem = base.rsplit(".", 1)[0]
                    key = stem_key(stem)
                    osv_id = want_keys.get(key)
                    if osv_id is not None and osv_id not in mapping:
                        mapping[osv_id] = {
                            "zip": z,
                            "member": name,
                            "size": sizes_all.get(osv_id),
                        }
        coverage = len(mapping) / max(1, len(want_keys))
        missing = [i for i in ids_df["osv_id"] if i not in mapping]
        summary[split] = {
            "zips": len(zips),
            "wanted": len(want_keys),
            "mapped": len(mapping),
            "coverage": round(coverage, 4),
        }
        log(f"[map] {split}: mapped {len(mapping)}/{len(want_keys)} ids (coverage {coverage:.3f})")
        if missing:
            log(f"[map] {split}: {len(missing)} unmapped, first: {missing[:10]}")
        zip_map[split] = mapping

    zip_map_path = OUT_ROOT / "zip_map.json"
    tmp = Path(str(zip_map_path) + ".tmp")
    tmp.write_text(json.dumps(zip_map), encoding="utf-8")
    os.replace(tmp, zip_map_path)
    log(f"[map] wrote {zip_map_path}")
    for split, s in summary.items():
        verdict = "PASS" if s["coverage"] >= MAP_COVERAGE_GATE else "REVIEW"
        log(f"[map] gate coverage >= {MAP_COVERAGE_GATE}: {split} {s['coverage']:.3f} -> {verdict}")
    return summary


# --------------------------------------------------------------------------
# Phase: extract
# --------------------------------------------------------------------------

def select_tasks(mapping, limit):
    """Group mapping by zip; with limit, round-robin one image per zip first."""
    by_zip = {}
    for osv_id, info in mapping.items():
        by_zip.setdefault(info["zip"], []).append((osv_id, info["member"], info.get("size")))
    if not limit:
        return by_zip
    selected = {}
    remaining = limit
    idx = 0
    keys = list(by_zip.keys())
    while remaining > 0:
        progressed = False
        for z in keys:
            if remaining <= 0:
                break
            lst = by_zip[z]
            if idx < len(lst):
                selected.setdefault(z, []).append(lst[idx])
                remaining -= 1
                progressed = True
        if not progressed:
            break
        idx += 1
    return selected


def extract_zip_task(split, zrel, members, ids_meta, args):
    out_dir = OUT_ROOT / split / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = OUT_ROOT / "cache" / "done" / f"{split}__{san(zrel)}.done"
    meta_path = OUT_ROOT / "cache" / "meta" / f"{split}__{san(zrel)}.csv"
    if done_marker.exists():
        try:
            marker_n = int(done_marker.read_text(encoding="utf-8").strip().rsplit("|", 1)[-1])
        except (OSError, ValueError):
            marker_n = None
        if marker_n == len(members):
            return {"zip": zrel, "ok": 0, "skip": 0, "fail": 0, "total": 0, "already": True,
                    "failures": [], "dry": False, "est_bytes": 0}

    if args.dry_run:
        est_bytes = 0
        for osv_id, member, size in members:
            est_bytes += int(size or 0)
        return {"zip": zrel, "ok": 0, "skip": 0, "fail": 0, "total": len(members),
                "already": False, "failures": [], "dry": True, "est_bytes": est_bytes}

    url = hf_url(zrel)
    last = None
    zf = None
    for attempt in range(3):
        try:
            zf, _rz = open_remote_zip(url)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    if zf is None:
        return {"zip": zrel, "ok": 0, "skip": 0, "fail": len(members), "total": len(members),
                "already": False, "failures": [(zrel, "", f"open: {last}")], "dry": False,
                "est_bytes": 0}

    ok = skip = 0
    failures = []
    rows = []
    try:
        for osv_id, member, size in members:
            lat, lon, town = ids_meta[osv_id]
            dest = out_dir / f"{osv_id}.jpg"
            try:
                info = zf.getinfo(member)
                expected = info.file_size
            except KeyError:
                expected = int(size or 0)
            if (
                dest.exists()
                and dest.stat().st_size > 0
                and (expected == 0 or dest.stat().st_size == expected)
            ):
                skip += 1
                rows.append((osv_id, lat, lon, town, zrel, member, expected, 1, 1))
                continue
            try:
                data = zf.read(member)
            except Exception as exc:  # noqa: BLE001
                failures.append((osv_id, member, f"read: {exc}"))
                continue
            if expected and len(data) != expected:
                failures.append((osv_id, member, f"size mismatch {len(data)} != {expected}"))
                continue
            if not data[:3] == b"\xff\xd8\xff":
                failures.append((osv_id, member, "not a JPEG (bad magic)"))
                continue
            tmp = Path(str(dest) + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            pil_ok = 1
            if args.verify_images:
                try:
                    from PIL import Image

                    with Image.open(dest) as im:
                        im.verify()
                except Exception as exc:  # noqa: BLE001
                    failures.append((osv_id, member, f"PIL verify: {exc}"))
                    try:
                        os.remove(dest)
                    except OSError:
                        pass
                    pil_ok = 0
                    continue
            ok += 1
            rows.append((osv_id, lat, lon, town, zrel, member, len(data), pil_ok, 0))
    finally:
        try:
            zf.close()
        except Exception:  # noqa: BLE001
            pass

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(meta_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write("osv_id,lat,lon,town,zip,member,bytes,pil_ok,skipped\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    os.replace(tmp, meta_path)

    if not failures and len(rows) == len(members):
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text(
            f"{datetime.now(timezone.utc).isoformat()}|{len(members)}", encoding="utf-8"
        )

    return {"zip": zrel, "ok": ok, "skip": skip, "fail": len(failures),
            "total": len(rows), "already": False, "failures": failures, "dry": False,
            "est_bytes": 0}


def phase_extract(args, splits):
    zip_map_path = OUT_ROOT / "zip_map.json"
    if not zip_map_path.exists():
        raise SystemExit(f"[fatal] missing {zip_map_path}; run `map` phase first")
    zip_map = json.loads(zip_map_path.read_text(encoding="utf-8"))

    all_failures = []
    for split in splits:
        mapping = zip_map.get(split, {})
        if not mapping:
            log(f"[extract] {split}: nothing mapped, skipping")
            continue
        ids_df = read_ids(split)
        ids_meta = {
            row.osv_id: (row.lat, row.lon, row.town)
            for row in ids_df.itertuples(index=False)
        }
        tasks = select_tasks(mapping, args.limit)
        total_imgs = sum(len(v) for v in tasks.values())
        total_bytes = sum(int(s or 0) for v in tasks.values() for (_i, _m, s) in v)
        log(
            f"[extract] {split}: {total_imgs} images across {len(tasks)} zips "
            f"(~{total_bytes / 1e6:.1f} MB)"
        )
        if args.dry_run:
            for zrel, members in sorted(tasks.items(), key=lambda kv: -len(kv[1])):
                est = sum(int(s or 0) for (_i, _m, s) in members)
                log(f"[extract] dry-run {zrel}: {len(members)} images (~{est / 1e6:.2f} MB)")
            log(f"[extract] dry-run complete; no image bytes downloaded")
            continue

        done_zips = 0
        done_imgs = 0
        ok_total = skip_total = fail_total = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(extract_zip_task, split, zrel, members, ids_meta, args): zrel
                for zrel, members in tasks.items()
            }
            for fut in as_completed(futures):
                zrel = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"[extract] FAILED {zrel}: {exc}")
                    all_failures.append((split, zrel, "", f"task: {exc}"))
                    continue
                done_zips += 1
                done_imgs += res["total"]
                ok_total += res["ok"]
                skip_total += res["skip"]
                fail_total += res["fail"]
                for f in res["failures"]:
                    all_failures.append((split,) + tuple(f))
                tag = "cached" if res["already"] else "done"
                log(
                    f"[extract] {split}: {zrel} {tag}: +{res['ok']} new, {res['skip']} "
                    f"skipped, {res['fail']} failed "
                    f"[{done_zips}/{len(tasks)} zips, {done_imgs}/{total_imgs} imgs, "
                    f"{time.time() - t0:.0f}s]"
                )
        log(
            f"[extract] {split}: finished {ok_total} new + {skip_total} skipped, "
            f"{fail_total} failures in {time.time() - t0:.0f}s"
        )

    if all_failures:
        fail_path = OUT_ROOT / "cache" / "failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_path, "a", encoding="utf-8", newline="") as fh:
            for row in all_failures:
                fh.write(",".join(str(v).replace(",", ";") for v in row) + "\n")
        log(f"[extract] {len(all_failures)} failures logged to {fail_path}")
    else:
        log("[extract] no failures")


# --------------------------------------------------------------------------
# Phase: verify
# --------------------------------------------------------------------------

def phase_verify(args, splits):
    meta_dir = OUT_ROOT / "cache" / "meta"
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "revision": REVISION,
        "bbox": KERALA_BBOX,
        "limited": bool(args.limit),
        "splits": {},
        "failures": [],
    }

    zip_map = {}
    zip_map_path = OUT_ROOT / "zip_map.json"
    if zip_map_path.exists():
        zip_map = json.loads(zip_map_path.read_text(encoding="utf-8"))

    for split in splits:
        ids_path = OUT_ROOT / "ids" / f"ids_{split}.csv"
        if not ids_path.exists():
            log(f"[verify] {split}: no ids file, skipping")
            continue
        ids_df = pd.read_csv(ids_path, dtype={"osv_id": str})
        expected_ids = set(ids_df["osv_id"])
        mapped_ids = set(zip_map.get(split, {}))

        frames = []
        if meta_dir.exists():
            for meta_path in sorted(meta_dir.glob(f"{split}__*.csv")):
                try:
                    frames.append(pd.read_csv(meta_path, dtype={"osv_id": str}))
                except Exception as exc:  # noqa: BLE001
                    log(f"[verify] bad meta {meta_path.name}: {exc}")
        if frames:
            meta = pd.concat(frames, ignore_index=True)
            meta = meta.drop_duplicates(subset="osv_id", keep="last")
        else:
            meta = pd.DataFrame(
                columns=["osv_id", "lat", "lon", "town", "zip", "member", "bytes",
                         "pil_ok", "skipped"]
            )

        images_dir = OUT_ROOT / split / "images"
        on_disk = {p.stem for p in images_dir.glob("*.jpg")} if images_dir.exists() else set()
        extracted = set(meta["osv_id"]) & on_disk

        # image sanity: size>0 for all, random-sample JPEG magic
        bad_magic = 0
        zero_size = 0
        sample = sorted(on_disk)
        sample = sample[:: max(1, len(sample) // 200)][:200]
        for osv_id in on_disk:
            p = images_dir / f"{osv_id}.jpg"
            try:
                if p.stat().st_size == 0:
                    zero_size += 1
            except OSError:
                pass
        for osv_id in sample:
            p = images_dir / f"{osv_id}.jpg"
            try:
                with open(p, "rb") as fh:
                    if fh.read(3) != b"\xff\xd8\xff":
                        bad_magic += 1
            except OSError:
                bad_magic += 1

        # final metadata CSVs
        meta_out = meta[meta["osv_id"].isin(on_disk)].sort_values("osv_id")
        meta_out.to_csv(OUT_ROOT / f"{split}_metadata.csv", index=False)

        town_counts = meta_out["town"].value_counts().to_dict() if len(meta_out) else {}
        cov = len(mapped_ids) / max(1, len(expected_ids))
        ratio = len(extracted) / max(1, len(mapped_ids)) if mapped_ids else 0.0
        missing = sorted(mapped_ids - extracted)

        split_report = {
            "ids_expected": len(expected_ids),
            "mapped": len(mapped_ids),
            "map_coverage": round(cov, 4),
            "extracted_on_disk": len(extracted),
            "extract_ratio": round(ratio, 4),
            "missing_sample": missing[:20],
            "zero_size_files": zero_size,
            "bad_magic_sample": bad_magic,
            "sample_checked": len(sample),
            "towns": {k: int(v) for k, v in town_counts.items()},
        }
        report["splits"][split] = split_report

        log(f"[verify] {split}: ids {len(expected_ids)}, mapped {len(mapped_ids)} "
            f"(cov {cov:.3f}), extracted {len(extracted)} (ratio {ratio:.3f})")
        log(f"[verify] {split}: towns {town_counts}")
        if zero_size:
            log(f"[verify] {split}: WARN {zero_size} zero-size files")
        if bad_magic:
            log(f"[verify] {split}: WARN {bad_magic}/{len(sample)} bad JPEG magic in sample")
        if missing and not args.limit:
            log(f"[verify] {split}: {len(missing)} mapped-but-missing, first: {missing[:10]}")

    # gates
    gates = {}
    train = report["splits"].get("train")
    if train:
        lo, hi = TRAIN_ID_GATE
        gates["train_ids_in_range"] = lo <= train["ids_expected"] <= hi
        gates["map_coverage_ok"] = train["map_coverage"] >= MAP_COVERAGE_GATE
        gates["extract_ratio_ok"] = train["extract_ratio"] >= EXTRACT_RATIO_GATE
    if args.limit:
        gates = {k: None for k in gates}  # limited smoke run: gates not meaningful
    report["gates"] = gates

    fail_path = OUT_ROOT / "cache" / "failures.csv"
    if fail_path.exists():
        report["failures"] = fail_path.read_text(encoding="utf-8").splitlines()[:100]

    report_path = OUT_ROOT / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, val in gates.items():
        log(f"[verify] gate {name}: {'PASS' if val else ('SKIP' if val is None else 'FAIL')}")
    log(f"[verify] wrote {report_path}")

    # attribution file for the harvested dataset
    attribution = OUT_ROOT / "README.md"
    if not attribution.exists():
        attribution.write_text(
            "# Kerala OSV-5M slice\n\n"
            f"Source: HuggingFace `{DATA_REPO}` @ `{REVISION}` (OSV-5M), "
            "license CC-BY-SA-4.0.\n"
            f"Selection: coordinate index `{INDEX_REPO}`, bbox "
            f"{KERALA_BBOX} (Kerala, India).\n"
            "Downloads are range-extracted from the remote image zips; see "
            "`scripts/extract_kerala_osv5m.py` and `extraction_report.json`.\n",
            encoding="utf-8",
        )
    return report


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=["ids", "map", "extract", "verify", "all"],
        help="which phase to run (default: all)",
    )
    parser.add_argument("--splits", default="train,test",
                        help="comma-separated splits (default: train,test)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap images per split for smoke runs (0 = no limit)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel workers for map/extract (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="extract: print the plan and estimated bytes, download nothing")
    parser.add_argument("--no-verify-images", dest="verify_images",
                        action="store_false", default=True,
                        help="skip full PIL verify of every extracted image")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"[run] phase={args.phase} splits={splits} limit={args.limit} "
        f"dry_run={args.dry_run} workers={args.workers}")

    if args.phase in ("ids", "all"):
        phase_ids(args, splits)
    if args.phase in ("map", "all"):
        phase_map(args, splits)
    if args.phase in ("extract", "all"):
        phase_extract(args, splits)
    if args.phase in ("verify", "all"):
        phase_verify(args, splits)
    log("[run] done")


if __name__ == "__main__":
    main()
