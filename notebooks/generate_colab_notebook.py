#!/usr/bin/env python3
"""
Run once to generate nmr_hsqc_colab.ipynb:
    python3 notebooks/generate_colab_notebook.py

Then upload nmr_hsqc_colab.ipynb to Google Colab,
set Runtime > Change runtime type > T4 GPU, and Run All.
"""

import json, os

cells = []

def code(src, **meta):
    cells.append({"cell_type": "code", "metadata": meta,
                   "outputs": [], "execution_count": None,
                   "source": src.lstrip("\n")})

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.lstrip("\n")})

# ─────────────────────────────────────────────────────────────
# 0 · SETUP
# ─────────────────────────────────────────────────────────────

md("""
# NMR ¹H-¹⁵N HSQC Chemical Shift Prediction
**Pipeline:** PDB → Custom Geometric Embeddings → GCN → (δ¹H, δ¹⁵N) per residue

**Steps before running:**
1. `Runtime > Change runtime type > T4 GPU`
2. `Ctrl+F9` (Run All)
3. Data + checkpoints are saved to Google Drive — safe to resume after session reset
""")

code("""
# ── GPU check ──────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("WARNING: No GPU — go to Runtime > Change runtime type > T4 GPU")
""")

code("""
# ── Mount Google Drive ─────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
print("Drive mounted at /content/drive")
""")

code("""
# ── Install packages ───────────────────────────────────────
import subprocess, sys, torch

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args, "-q"], check=True)

pip("torch-geometric")

# Optional PyG C++ extensions (speed up sparse ops)
tv  = torch.__version__.split("+")[0]
cv  = torch.version.cuda.replace(".", "") if torch.cuda.is_available() else "cpu"
pip("torch-scatter", "torch-sparse",
    "-f", f"https://data.pyg.org/whl/torch-{tv}+cu{cv}.html")

pip("biopython", "pynmrstar", "tqdm", "scipy", "seaborn")
print("All packages installed.")
""")

code("""
# ── Configuration — edit this cell to tune the run ────────
CFG = dict(
    # Paths (all under your Drive)
    drive_root   = "/content/drive/MyDrive/nmr_hsqc_predictor",

    # Data
    max_entries  = 500,           # BMRB entries to download

    # Geometric embedding
    space_dist   = 5.0,           # Å, through-space neighbor cutoff
    seq_range    = 3,             # i±3 sequence neighbors
    max_nb       = 128,           # max neighbors per residue
    bb_cutoff    = 1.0,           # backbone RMSD filter (NMR ensembles)
    nb_cutoff    = 1.5,           # neighbor atom RMSD filter

    # Graph edges
    edge_dist    = 8.0,           # Å, spatial edge cutoff
    edge_seq     = 3,             # sequence edge range

    # Training
    node_feat_dim = 1792,         # 128 × 14 (fixed by embedding design)
    hidden_dim    = 256,
    num_layers    = 4,
    batch_size    = 32,
    epochs        = 200,
    lr            = 1e-3,
    weight_decay  = 1e-4,
    patience      = 20,
    seed          = 42,
)

# Derived paths
import os
R = CFG["drive_root"]
BMRB_DIR   = f"{R}/data/raw/bmrb"
PDB_DIR    = f"{R}/data/raw/pdb"
MANIFEST   = f"{R}/data/raw/manifest.json"
GRAPHS_PKL = f"{R}/data/processed/graphs.pkl"
SPLITS_DIR = f"{R}/data/splits"
CKPT_GCN   = f"{R}/checkpoints/best_gcn.pt"
CKPT_MLP   = f"{R}/checkpoints/best_mlp.pt"
OUTPUT_DIR = f"{R}/output"

for d in [BMRB_DIR, PDB_DIR, f"{R}/data/processed",
          SPLITS_DIR, f"{R}/checkpoints", OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print("Directories ready under", R)
""")

code("""
# ── Imports ────────────────────────────────────────────────
import json, os, pickle, time, logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(CFG["seed"])
print(f"Using device: {DEVICE}")
""")

# ─────────────────────────────────────────────────────────────
# 1 · DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────

md("## 1 · Data Download\nFetches ~500 BMRB NMR-STAR files + linked PDB structures. Resumable.")

code("""
BMRB_API = "https://api.bmrb.io/v2"
BMRB_FTP = "https://bmrb.io/ftp/pub/bmrb/entry_directories"
RCSB_URL = "https://files.rcsb.org/download"

def get_mapping():
    r = requests.get(f"{BMRB_API}/mappings/bmrb/pdb", timeout=60)
    r.raise_for_status()
    return {str(k): [p.upper() for p in v]
            for k, v in r.json().items() if v}

def dl_bmrb(bmrb_id):
    fname = f"bmr{bmrb_id}_3.str"
    dest  = os.path.join(BMRB_DIR, fname)
    if os.path.exists(dest):
        return dest
    try:
        r = requests.get(f"{BMRB_FTP}/bmr{bmrb_id}/{fname}", timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f: f.write(r.content)
        return dest
    except Exception:
        return None

def dl_pdb(pdb_id):
    pid  = pdb_id.lower()
    dest = os.path.join(PDB_DIR, f"{pid}.pdb")
    if os.path.exists(dest):
        return dest
    try:
        r = requests.get(f"{RCSB_URL}/{pid}.pdb", timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f: f.write(r.content)
        return dest
    except Exception:
        return None

# ── Resume from existing manifest ─────────────────────────
manifest = []
if os.path.exists(MANIFEST):
    with open(MANIFEST) as f: manifest = json.load(f)
    ok = sum(1 for r in manifest if r.get("ok"))
    print(f"Resuming: {len(manifest)} processed, {ok} valid pairs so far")
already = {r["bmrb_id"] for r in manifest}

print("Fetching BMRB→PDB mapping …")
mapping = get_mapping()
print(f"  {len(mapping)} entries with linked PDB")

candidates = sorted(mapping.keys(), key=lambda x: int(x))
downloaded = sum(1 for r in manifest if r.get("ok"))

for bmrb_id in tqdm(candidates, desc="Downloading"):
    if downloaded >= CFG["max_entries"]:
        break
    if bmrb_id in already:
        if any(r["bmrb_id"] == bmrb_id and r.get("ok") for r in manifest):
            downloaded += 1
        continue

    pdb_id = mapping[bmrb_id][0]
    bp = dl_bmrb(bmrb_id)
    pp = dl_pdb(pdb_id)
    ok = bp is not None and pp is not None

    manifest.append({"bmrb_id": bmrb_id, "pdb_id": pdb_id,
                     "bmrb_path": bp, "pdb_path": pp, "ok": ok})
    if ok:
        downloaded += 1
    with open(MANIFEST, "w") as f: json.dump(manifest, f)
    time.sleep(0.3)

valid = [r for r in manifest if r.get("ok")]
print(f"\\nDone: {len(valid)} valid BMRB+PDB pairs")
""")

# ─────────────────────────────────────────────────────────────
# 2 · PARSING
# ─────────────────────────────────────────────────────────────

md("## 2 · Parsing\nExtract amide shifts from NMR-STAR; parse atom coordinates from PDB.")

code("""
# ── BMRB parser ────────────────────────────────────────────
import pynmrstar

AMIDE_H = {"H", "HN", "H1"}

def parse_hsqc_shifts(star_file):
    \"\"\"Returns {(chain, seq_id, res_name): (dH, dN)}\"\"\n    try:
        entry = pynmrstar.Entry.from_file(star_file)
    except Exception:
        return {}

    loops = []
    try:
        loops = entry.get_loops_by_category("Atom_chem_shift")
    except AttributeError:
        for sf in entry:
            for lp in sf:
                if hasattr(lp, "tags") and any(
                    "atom_chem_shift" in t.lower() for t in lp.tags
                ):
                    loops.append(lp)

    result = {}
    for loop in loops:
        try:
            tags = [t.split(".")[-1].lower() for t in loop.tags]
            needed = {"atom_id", "val", "seq_id", "comp_id"}
            if not needed.issubset(tags):
                continue
            idx = {t: i for i, t in enumerate(tags)}
            shifts = {}

            for row in loop.data:
                atom = str(row[idx["atom_id"]]).upper()
                val_raw = row[idx["val"]]
                comp    = str(row[idx["comp_id"]]).upper()
                if comp == "PRO" or val_raw in (".", "?", "", None):
                    continue
                try:
                    val = float(val_raw)
                    seq = int(row[idx["seq_id"]])
                except (ValueError, TypeError):
                    continue

                chain = "1"
                if "entity_assembly_id" in idx:
                    chain = str(row[idx["entity_assembly_id"]])

                key = (chain, seq, comp)
                shifts.setdefault(key, {})

                is_h = atom in AMIDE_H
                is_n = atom == "N"

                if "atom_isotope_number" in idx:
                    iso_raw = row[idx["atom_isotope_number"]]
                    if iso_raw not in (".", "?", None):
                        try:
                            iso = int(iso_raw)
                            if is_h and iso != 1:  is_h = False
                            if is_n and iso != 15: is_n = False
                        except ValueError:
                            pass

                if is_h and "H" not in shifts[key]: shifts[key]["H"] = val
                if is_n and "N" not in shifts[key]: shifts[key]["N"] = val

            for key, d in shifts.items():
                if "H" in d and "N" in d:
                    result[key] = (d["H"], d["N"])
        except Exception:
            continue
    return result
""")

code("""
# ── PDB parser (handles NMR ensembles) ─────────────────────
from Bio.PDB import PDBParser as _PDBParser

_parser = _PDBParser(QUIET=True)
BB_ATOMS = {"N", "CA", "C", "O"}

def load_residue_atoms(pdb_file, bb_cutoff=1.0):
    \"\"\"
    Returns (residue_atoms, flexible_ok).
    residue_atoms : {(chain, seq, res): [{name, coord, coord_std, element, rmsd}]}
    flexible_ok   : set of keys passing RMSD filter (None for single-model)
    \"\"\"
    struct   = _parser.get_structure("X", pdb_file)
    n_models = len(list(struct.get_models()))

    if n_models > 1:
        # NMR ensemble: average + std across models
        coords_by_atom = defaultdict(list)
        for model in struct.get_models():
            for chain in model:
                for res in chain:
                    hf, seq_id, _ = res.id
                    if hf.strip(): continue
                    for atom in res.get_atoms():
                        key = (chain.id, seq_id, res.resname.strip(), atom.name)
                        coords_by_atom[key].append(atom.coord.copy())

        stats = {}
        for key, cl in coords_by_atom.items():
            arr  = np.array(cl, dtype=np.float32)
            mean = arr.mean(0)
            std  = arr.std(0)
            rmsd = float(np.sqrt(((arr - mean)**2).sum(1).mean()))
            stats[key] = {"mean": mean, "std": std, "rmsd": rmsd}

        # Group into residues
        res_atoms = defaultdict(list)
        for (ch, sq, rn, an), s in stats.items():
            res_atoms[(ch, sq, rn)].append({
                "name": an, "coord": s["mean"], "coord_std": s["std"],
                "element": an[0].upper(), "rmsd": s["rmsd"]
            })

        # Backbone RMSD filter
        bb_rmsds = defaultdict(list)
        for (ch, sq, rn, an), s in stats.items():
            if an in BB_ATOMS:
                bb_rmsds[(ch, sq, rn)].append(s["rmsd"])
        flex_ok = {k for k, v in bb_rmsds.items()
                   if v and np.mean(v) < bb_cutoff}

        return dict(res_atoms), flex_ok

    else:
        # X-ray: single model, zero std
        res_atoms = {}
        for model in struct.get_models():
            for chain in model:
                for res in chain:
                    hf, seq_id, _ = res.id
                    if hf.strip(): continue
                    key = (chain.id, seq_id, res.resname.strip())
                    res_atoms[key] = [{
                        "name": a.name, "coord": a.coord.copy(),
                        "coord_std": np.zeros(3, np.float32),
                        "element": (a.element or a.name[0]).upper(),
                        "rmsd": 0.0
                    } for a in res.get_atoms()]
            break
        return res_atoms, None
""")

# ─────────────────────────────────────────────────────────────
# 3 · FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

md("## 3 · Custom Geometric Embeddings\nLocal coordinate frame per residue + 128 neighbor atoms → 1792-dim vector.")

code("""
ELEMENTS = ["C", "N", "O", "S", "H"]      # +1 "other" bin
FPB = 3 + 3 + 1 + 1 + len(ELEMENTS) + 1   # features per neighbor = 14

def one_hot_elem(e):
    v = [0.0] * (len(ELEMENTS) + 1)
    c = e.strip().upper()[:1]
    v[ELEMENTS.index(c) if c in ELEMENTS else -1] = 1.0
    return v

def build_local_frame(N, Ca, C):
    \"\"\"Returns 3×3 rotation R: R@(Ca-N)→+X, R@(C-N) in +Y half-plane.\"\"\n    ca, c = Ca - N, C - N

    ca_xy = np.array([ca[0], ca[1], 0.])
    nxy = np.linalg.norm(ca_xy)
    Rz = np.eye(3) if nxy < 1e-8 else np.array([
        [ ca_xy[0]/nxy, -ca_xy[1]/nxy, 0],
        [ ca_xy[1]/nxy,  ca_xy[0]/nxy, 0],
        [0, 0, 1]], dtype=np.float64)

    ca1 = Rz @ ca
    nxz = np.sqrt(ca1[0]**2 + ca1[2]**2)
    Ry = np.eye(3) if nxz < 1e-8 else np.array([
        [ ca1[0]/nxz, 0, ca1[2]/nxz],
        [0, 1, 0],
        [-ca1[2]/nxz, 0, ca1[0]/nxz]], dtype=np.float64)

    R1 = Ry @ Rz
    c1  = R1 @ c
    nyz = np.sqrt(c1[1]**2 + c1[2]**2)
    Rx  = np.eye(3) if nyz < 1e-8 else np.array([
        [1, 0, 0],
        [0,  c1[1]/nyz, -c1[2]/nyz],
        [0,  c1[2]/nyz,  c1[1]/nyz]], dtype=np.float64)
    return Rx @ R1

def _get(atoms, name):
    for a in atoms:
        if a["name"] == name: return a["coord"]
    return None

def embed_residue(key, res_atoms, seq_order, cfg):
    atoms = res_atoms.get(key)
    if not atoms: return None
    N, Ca, C = _get(atoms,"N"), _get(atoms,"CA"), _get(atoms,"C")
    if N is None or Ca is None or C is None: return None

    R = build_local_frame(N, Ca, C)
    ch, sq, _ = key
    my_i = next((i for i,k in enumerate(seq_order) if k==key), None)

    rows, seen = [], set()
    for ok, oa in res_atoms.items():
        if ok == key: continue
        oc, osq, _ = ok
        if oc != ch: continue
        oi = next((i for i,k in enumerate(seq_order) if k==ok), None)
        in_seq = int(my_i is not None and oi is not None and abs(oi-my_i) <= cfg["seq_range"])

        for a in oa:
            ak = (ok, a["name"])
            if ak in seen: continue
            if a.get("rmsd", 0) > cfg["nb_cutoff"]: continue
            d = np.linalg.norm(a["coord"] - N)
            in_sp = int(d <= cfg["space_dist"])
            if not in_sp and not in_seq: continue
            seen.add(ak)
            rows.append((d, a["coord"], a.get("coord_std", np.zeros(3)),
                         a["element"], float(in_sp), float(in_seq)))

    rows.sort(key=lambda x: x[0])
    feat = []
    for _, coord, std, elem, isp, iseq in rows[:cfg["max_nb"]]:
        lc = R @ (coord - N)
        ls = R @ std
        feat.extend([*lc, *ls, isp, iseq, *one_hot_elem(elem)])
    while len(feat) < cfg["max_nb"] * FPB:
        feat.extend([0.0] * FPB)
    return np.array(feat[:cfg["max_nb"]*FPB], dtype=np.float32)
""")

code("""
# ── Build PyG graphs for all proteins ──────────────────────
# Cached to Drive. Delete graphs.pkl to rebuild from scratch.

def build_edges(node_keys, res_atoms, cfg):
    n = len(node_keys)
    nc = [_get(res_atoms.get(k,[]), "N") for k in node_keys]
    src, dst = [], []
    for i in range(n):
        ci, si, _ = node_keys[i]
        for j in range(i+1, n):
            cj, sj, _ = node_keys[j]
            conn = (ci==cj and abs(si-sj) <= cfg["edge_seq"])
            if not conn and nc[i] is not None and nc[j] is not None:
                conn = np.linalg.norm(nc[i]-nc[j]) <= cfg["edge_dist"]
            if conn:
                src += [i,j]; dst += [j,i]
    if not src:
        return torch.zeros((2,0), dtype=torch.long)
    return torch.tensor([src,dst], dtype=torch.long)

def process_protein(entry, cfg):
    shifts = parse_hsqc_shifts(entry["bmrb_path"])
    if not shifts: return None

    res_atoms, flex_ok = load_residue_atoms(entry["pdb_path"], cfg["bb_cutoff"])
    target = [k for k in shifts if k in res_atoms]
    if not target: return None

    # Build per-chain seq_order
    chain_ord = defaultdict(list)
    for k in sorted(res_atoms, key=lambda x: x[1]):
        chain_ord[k[0]].append(k)
    seq_order = [k for v in chain_ord.values() for k in v]

    feats = {}
    for k in target:
        if flex_ok is not None and k not in flex_ok: continue
        v = embed_residue(k, res_atoms, seq_order, cfg)
        if v is not None: feats[k] = v

    valid = [k for k in target if k in feats]
    if len(valid) < 3: return None

    X = np.stack([feats[k] for k in valid])
    y = np.array([shifts[k] for k in valid], dtype=np.float32)
    ei = build_edges(valid, res_atoms, cfg)
    return Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=ei,
                y=torch.tensor(y, dtype=torch.float32),
                num_nodes=len(valid))

if os.path.exists(GRAPHS_PKL):
    print("Loading cached graphs …")
    with open(GRAPHS_PKL, "rb") as f: graph_map = pickle.load(f)
    print(f"  {len(graph_map)} graphs loaded")
else:
    valid_entries = [r for r in manifest if r.get("ok")]
    graph_map = {}
    skip = 0
    for entry in tqdm(valid_entries, desc="Processing proteins"):
        pid = entry["bmrb_id"]
        try:
            data = process_protein(entry, CFG)
            if data: graph_map[pid] = data
            else: skip += 1
        except Exception as e:
            skip += 1
    with open(GRAPHS_PKL, "wb") as f: pickle.dump(graph_map, f)
    print(f"Cached {len(graph_map)} graphs  ({skip} skipped) → {GRAPHS_PKL}")

print(f"Total proteins with graphs: {len(graph_map)}")
""")

code("""
# ── Protein-level 70/20/10 split ──────────────────────────
pids = sorted(graph_map.keys())
train_ids, tmp = train_test_split(pids, test_size=0.30, random_state=CFG["seed"])
val_ids, test_ids = train_test_split(tmp,  test_size=0.333, random_state=CFG["seed"])

for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    Path(SPLITS_DIR).mkdir(exist_ok=True)
    with open(f"{SPLITS_DIR}/{name}.txt", "w") as f:
        f.write("\\n".join(ids))

class HSQCDataset:
    def __init__(self, ids): self._d = [graph_map[i] for i in ids if i in graph_map]
    def __len__(self):       return len(self._d)
    def __getitem__(self, i): return self._d[i]

from torch_geometric.data import Batch

train_ds = HSQCDataset(train_ids)
val_ds   = HSQCDataset(val_ids)
test_ds  = HSQCDataset(test_ids)

print(f"Split — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)} proteins")

BS = CFG["batch_size"]
train_loader = GeoLoader(train_ds, batch_size=BS, shuffle=True)
val_loader   = GeoLoader(val_ds,   batch_size=BS, shuffle=False)
test_loader  = GeoLoader(test_ds,  batch_size=BS, shuffle=False)
""")

# ─────────────────────────────────────────────────────────────
# 4 · MODELS
# ─────────────────────────────────────────────────────────────

md("## 4 · Models")

code("""
from torch_geometric.nn import GCNConv

class NMRShiftGCN(nn.Module):
    def __init__(self, feat_dim, hidden=256, layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(feat_dim, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.drop = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 2)
        )
    def forward(self, x, edge_index, batch=None):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(norm(F.relu(conv(x, edge_index))),
                          p=self.drop, training=self.training)
        return self.head(x)


class NMRShiftMLP(nn.Module):
    def __init__(self, feat_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
    def forward(self, x, edge_index=None, batch=None):
        return self.net(x)

F_DIM = CFG["node_feat_dim"]
H_DIM = CFG["hidden_dim"]
gcn_model = NMRShiftGCN(F_DIM, H_DIM, CFG["num_layers"]).to(DEVICE)
mlp_model = NMRShiftMLP(F_DIM, H_DIM).to(DEVICE)
print(f"GCN params: {sum(p.numel() for p in gcn_model.parameters()):,}")
print(f"MLP params: {sum(p.numel() for p in mlp_model.parameters()):,}")
""")

# ─────────────────────────────────────────────────────────────
# 5 · TRAINING
# ─────────────────────────────────────────────────────────────

md("## 5 · Training")

code("""
def eval_mae(model, loader):
    model.eval()
    totH = totN = n = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            p = model(b.x, b.edge_index, b.batch)
            d = (p - b.y).abs()
            totH += d[:,0].sum().item()
            totN += d[:,1].sum().item()
            n    += d.shape[0]
    return (totH/n, totN/n) if n else (float("inf"), float("inf"))


def train_model(model, name, ckpt_path, epochs=None, patience=None):
    epochs  = epochs  or CFG["epochs"]
    patience= patience or CFG["patience"]

    opt  = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sch  = CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss()

    best, no_imp = float("inf"), 0
    hist = []

    for ep in range(1, epochs+1):
        model.train()
        loss_sum, nb = 0., 0
        for b in train_loader:
            b = b.to(DEVICE)
            pred = model(b.x, b.edge_index, b.batch)
            loss = crit(pred, b.y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        sch.step()

        mH, mN = eval_mae(model, val_loader)
        combined = mH + mN
        hist.append((ep, loss_sum/nb, mH, mN))

        if ep % 20 == 0 or ep == 1:
            print(f"[{name}] ep {ep:3d} | loss={loss_sum/nb:.4f} | "
                  f"val MAE ¹H={mH:.3f} ¹⁵N={mN:.3f} ppm")

        if combined < best:
            best = combined
            torch.save(model.state_dict(), ckpt_path)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stop at epoch {ep}")
                break

    print(f"[{name}] Best val MAE sum = {best:.4f}  →  {ckpt_path}")
    return hist
""")

code("""
# ── Train GCN ─────────────────────────────────────────────
gcn_hist = train_model(gcn_model, "GCN", CKPT_GCN)
""")

code("""
# ── Train MLP baseline ────────────────────────────────────
mlp_hist = train_model(mlp_model, "MLP", CKPT_MLP)
""")

# ─────────────────────────────────────────────────────────────
# 6 · EVALUATION
# ─────────────────────────────────────────────────────────────

md("## 6 · Evaluation\nLoads best checkpoints; reports MAE + Pearson r; scatter plots saved to Drive.")

code("""
def collect(model, ckpt, loader):
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            preds.append(model(b.x, b.edge_index, b.batch).cpu().numpy())
            trues.append(b.y.cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)

TARGETS = {"mae_H": ("<", 0.3), "mae_N": ("<", 2.0),
           "r_H":   (">", 0.90), "r_N":  (">", 0.90)}

def report(name, y_true, y_pred):
    mH = float(np.mean(np.abs(y_pred[:,0]-y_true[:,0])))
    mN = float(np.mean(np.abs(y_pred[:,1]-y_true[:,1])))
    rH = float(pearsonr(y_true[:,0], y_pred[:,0])[0])
    rN = float(pearsonr(y_true[:,1], y_pred[:,1])[0])
    m  = {"mae_H": mH, "mae_N": mN, "r_H": rH, "r_N": rN}
    print(f"\\n=== {name} Test Results ===")
    for k, (op, thr) in TARGETS.items():
        ok = m[k] < thr if op=="<" else m[k] > thr
        print(f"  {k:<8}: {m[k]:.4f}  [{op}{thr}]  {'PASS' if ok else 'FAIL'}")
    return m, y_true, y_pred

def scatter(name, y_true, y_pred, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, lbl, thr_mae in [(axes[0],0,"¹H",0.3),(axes[1],1,"¹⁵N",2.0)]:
        xt, xp = y_true[:,col], y_pred[:,col]
        ax.scatter(xt, xp, alpha=0.25, s=6, rasterized=True)
        lo = min(xt.min(),xp.min())-.5; hi = max(xt.max(),xp.max())+.5
        ax.plot([lo,hi],[lo,hi],"k--",lw=1)
        ax.set(xlim=[lo,hi], ylim=[lo,hi],
               xlabel=f"Exp δ{lbl} (ppm)", ylabel=f"Pred δ{lbl} (ppm)")
        key = "H" if col==0 else "N"
        ax.set_title(f"{name} δ{lbl}  MAE={metrics[f'mae_{key}']:.3f}  r={metrics[f'r_{key}']:.3f}")
    fig.suptitle(name, fontsize=13)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/scatter_{name.lower()}.png"
    fig.savefig(path, dpi=150); plt.show()
    print(f"Saved: {path}")

gcn_metrics, _, _ = report("GCN", yt_gcn, yp_gcn)
scatter("GCN", yt_gcn, yp_gcn, gcn_metrics)

yt_mlp, yp_mlp = collect(mlp_model, CKPT_MLP, test_loader)
mlp_metrics, _, _ = report("MLP", yt_mlp, yp_mlp)
scatter("MLP", yt_mlp, yp_mlp, mlp_metrics)
""")

code("""
# ── Training loss curves ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, hist, name in [(axes[0], gcn_hist, "GCN"), (axes[1], mlp_hist, "MLP")]:
    eps  = [h[0] for h in hist]
    maes = [h[2]+h[3] for h in hist]   # combined val MAE (H + N)
    ax.plot(eps, maes)
    ax.set(xlabel="Epoch", ylabel="Val MAE ¹H+¹⁵N (ppm)", title=f"{name} training")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
plt.show()
""")

# ─────────────────────────────────────────────────────────────
# 7 · INFERENCE
# ─────────────────────────────────────────────────────────────

md("## 7 · Inference on a New PDB\nUpload any `.pdb` to Colab and predict its HSQC spectrum.")

code("""
def predict_pdb(pdb_file, model):
    \"\"\"Returns DataFrame: chain, seq_id, res_name, pred_1H, pred_15N\"\"\n    res_atoms, flex_ok = load_residue_atoms(pdb_file, CFG["bb_cutoff"])
    if not res_atoms:
        raise ValueError(f"No residues parsed from {pdb_file}")

    chain_ord = defaultdict(list)
    for k in sorted(res_atoms, key=lambda x: x[1]):
        chain_ord[k[0]].append(k)
    seq_order = [k for v in chain_ord.values() for k in v]

    feats = {}
    for k in seq_order:
        if flex_ok is not None and k not in flex_ok: continue
        v = embed_residue(k, res_atoms, seq_order, CFG)
        if v is not None: feats[k] = v

    valid = [k for k in seq_order if k in feats]
    if not valid:
        raise ValueError("No residues survived embedding")

    X  = torch.tensor(np.stack([feats[k] for k in valid]), dtype=torch.float32).to(DEVICE)
    ei = build_edges(valid, res_atoms, CFG).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(X, ei).cpu().numpy()

    return pd.DataFrame([{
        "chain": ch, "seq_id": sq, "res_name": rn,
        "pred_1H": round(float(pred[i,0]),4),
        "pred_15N": round(float(pred[i,1]),4)
    } for i, (ch,sq,rn) in enumerate(valid)])


# ── Example: predict first test protein ───────────────────
ex_pid  = test_ids[0]
ex_pdb  = next(r["pdb_path"] for r in manifest if r["bmrb_id"] == ex_pid)
gcn_model.load_state_dict(torch.load(CKPT_GCN, map_location=DEVICE))

df = predict_pdb(ex_pdb, gcn_model)
print(f"Predictions for BMRB {ex_pid} ({ex_pdb}):")
print(df.head(10).to_string(index=False))

out_csv = f"{OUTPUT_DIR}/pred_{ex_pid}.csv"
df.to_csv(out_csv, index=False)
print(f"\\nFull predictions saved to {out_csv}")
""")

code("""
# ── Predict any uploaded PDB ────────────────────────────────
# Upload a PDB via the Colab file picker, then set MY_PDB below.
# from google.colab import files
# uploaded = files.upload()
# MY_PDB = list(uploaded.keys())[0]

# Uncomment below after uploading:
# df_custom = predict_pdb(MY_PDB, gcn_model)
# print(df_custom.to_string(index=False))
# df_custom.to_csv(f"{OUTPUT_DIR}/pred_custom.csv", index=False)
print("Uncomment lines above after uploading your own PDB file.")
""")

# ─────────────────────────────────────────────────────────────
# WRITE NOTEBOOK
# ─────────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nmr_hsqc_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\nNotebook written to: {out}")
print("Next steps:")
print("  1. Run:  python3 notebooks/generate_colab_notebook.py")
print("  2. Upload nmr_hsqc_colab.ipynb to colab.research.google.com")
print("  3. Runtime > Change runtime type > T4 GPU")
print("  4. Ctrl+F9  (Run All)")
