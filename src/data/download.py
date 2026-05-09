"""
BMRB + PDB batch downloader.

Usage:
    python src/data/download.py --max-entries 500 --out-dir data/raw
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

BMRB_API = "https://api.bmrb.io/v2"
BMRB_FTP = "https://bmrb.io/ftp/pub/bmrb/entry_directories"
RCSB_URL = "https://files.rcsb.org/download"


def get_bmrb_pdb_mapping() -> dict[str, list[str]]:
    """Fetch bulk BMRB → PDB mapping from the BMRB API.

    Returns { bmrb_id: [pdb_id, ...] }
    """
    url = f"{BMRB_API}/mappings/bmrb/pdb"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    # Response shape: { "1": ["1ABC", ...], "2": [], ... }
    return {str(k): [p.upper() for p in v] for k, v in raw.items() if v}


def download_bmrb_star(bmrb_id: str, dest_dir: str) -> str | None:
    """Download NMR-STAR 3 file. Returns local path or None on failure."""
    fname = f"bmr{bmrb_id}_3.str"
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        return dest
    url = f"{BMRB_FTP}/bmr{bmrb_id}/{fname}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(resp.content)
        return dest
    except requests.HTTPError:
        return None


def download_pdb(pdb_id: str, dest_dir: str) -> str | None:
    """Download PDB file from RCSB. Returns local path or None on failure."""
    pdb_id = pdb_id.lower()
    dest = os.path.join(dest_dir, f"{pdb_id}.pdb")
    if os.path.exists(dest):
        return dest
    url = f"{RCSB_URL}/{pdb_id}.pdb"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(resp.content)
        return dest
    except requests.HTTPError:
        return None


def main(max_entries: int, bmrb_dir: str, pdb_dir: str, manifest_path: str):
    Path(bmrb_dir).mkdir(parents=True, exist_ok=True)
    Path(pdb_dir).mkdir(parents=True, exist_ok=True)

    print("Fetching BMRB → PDB mapping …")
    mapping = get_bmrb_pdb_mapping()
    print(f"  {len(mapping)} BMRB entries with at least one linked PDB")

    # Load existing manifest so interrupted runs are resumable
    manifest: list[dict] = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    already = {r["bmrb_id"] for r in manifest}

    candidates = sorted(mapping.keys(), key=lambda x: int(x))
    downloaded = len([r for r in manifest if r.get("ok")])

    for bmrb_id in tqdm(candidates, desc="Downloading"):
        if downloaded >= max_entries:
            break
        if bmrb_id in already:
            if any(r["bmrb_id"] == bmrb_id and r.get("ok") for r in manifest):
                downloaded += 1
            continue

        pdb_id = mapping[bmrb_id][0]  # take first linked PDB

        bmrb_path = download_bmrb_star(bmrb_id, bmrb_dir)
        pdb_path = download_pdb(pdb_id, pdb_dir)
        ok = bmrb_path is not None and pdb_path is not None

        manifest.append({
            "bmrb_id": bmrb_id,
            "pdb_id": pdb_id,
            "bmrb_path": bmrb_path,
            "pdb_path": pdb_path,
            "ok": ok,
        })

        if ok:
            downloaded += 1

        # Save incrementally
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        time.sleep(0.3)  # be polite to servers

    ok_count = sum(1 for r in manifest if r.get("ok"))
    print(f"\nDone. {ok_count} valid pairs saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-entries", type=int, default=500)
    parser.add_argument("--bmrb-dir", default="data/raw/bmrb")
    parser.add_argument("--pdb-dir", default="data/raw/pdb")
    parser.add_argument("--manifest", default="data/raw/manifest.json")
    args = parser.parse_args()

    main(args.max_entries, args.bmrb_dir, args.pdb_dir, args.manifest)
