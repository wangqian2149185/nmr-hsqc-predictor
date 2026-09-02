from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa


AA_ORDER_INFERENCE = [
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y",
]

AA3_TO_1_INFERENCE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def load_inference_chains(structure_path):
    structure_path = Path(structure_path)

    if structure_path.suffix.lower() in {
        ".cif",
        ".mmcif",
    }:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        structure_path.stem,
        str(structure_path),
    )

    model = next(structure.get_models())
    chains = {}

    for chain in model:
        residues = []

        for residue in chain:
            if not is_aa(residue, standard=True):
                continue

            comp_id = residue.get_resname().upper()

            if comp_id not in AA3_TO_1_INFERENCE:
                continue

            residues.append(residue)

        if residues:
            chains[str(chain.id)] = residues

    return chains


def inference_coord(residue, atom_name):
    if atom_name not in residue:
        return None

    return np.asarray(
        residue[atom_name].coord,
        dtype=float,
    )


def inference_dihedral(p0, p1, p2, p3):
    p0, p1, p2, p3 = [
        np.asarray(point, dtype=float)
        for point in [p0, p1, p2, p3]
    ]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    norm = np.linalg.norm(b1)

    if norm < 1e-8:
        return np.nan

    b1 = b1 / norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    if (
        np.linalg.norm(v) < 1e-8
        or np.linalg.norm(w) < 1e-8
    ):
        return np.nan

    return float(
        np.arctan2(
            np.dot(
                np.cross(b1, v),
                w,
            ),
            np.dot(v, w),
        )
    )


def inference_phi_psi(residues, index):
    residue = residues[index]

    n = inference_coord(residue, "N")
    ca = inference_coord(residue, "CA")
    c = inference_coord(residue, "C")

    phi = np.nan
    psi = np.nan

    if n is None or ca is None or c is None:
        return phi, psi

    if index > 0:
        previous_c = inference_coord(
            residues[index - 1],
            "C",
        )

        if previous_c is not None:
            phi = inference_dihedral(
                previous_c,
                n,
                ca,
                c,
            )

    if index + 1 < len(residues):
        next_n = inference_coord(
            residues[index + 1],
            "N",
        )

        if next_n is not None:
            psi = inference_dihedral(
                n,
                ca,
                c,
                next_n,
            )

    return phi, psi


def one_hot_residue(prefix, aa_code):
    return {
        f"{prefix}{aa}": float(aa_code == aa)
        for aa in AA_ORDER_INFERENCE
    }


def build_pdb_feature_frame(
    structure_path,
    chain_id,
    include_proline=False,
):
    chains = load_inference_chains(structure_path)
    chain_id = str(chain_id)

    if chain_id not in chains:
        raise KeyError(
            f"Chain {chain_id!r} was not found. "
            f"Available chains: {list(chains)}"
        )

    target_residues = chains[chain_id]
    chain_length = len(target_residues)

    if chain_length < 2:
        raise ValueError(
            "The target chain contains fewer than two standard residues."
        )

    environment = []

    for environment_chain_id, residues in chains.items():
        for environment_index, residue in enumerate(residues):
            environment.append({
                "chain_id": str(environment_chain_id),
                "chain_index": environment_index,
                "ca": inference_coord(residue, "CA"),
                "o": inference_coord(residue, "O"),
            })

    feature_rows = []

    for index, residue in enumerate(target_residues):
        comp_id = residue.get_resname().upper()
        aa_code = AA3_TO_1_INFERENCE[comp_id]

        if aa_code == "P" and not include_proline:
            continue

        previous_aa = (
            AA3_TO_1_INFERENCE[
                target_residues[index - 1]
                .get_resname()
                .upper()
            ]
            if index > 0
            else None
        )

        next_aa = (
            AA3_TO_1_INFERENCE[
                target_residues[index + 1]
                .get_resname()
                .upper()
            ]
            if index + 1 < chain_length
            else None
        )

        phi, psi = inference_phi_psi(
            target_residues,
            index,
        )

        phi_mask = float(np.isfinite(phi))
        psi_mask = float(np.isfinite(psi))

        target_ca = inference_coord(residue, "CA")
        target_n = inference_coord(residue, "N")

        contact_counts = {
            6: np.nan,
            8: np.nan,
            10: np.nan,
        }

        if target_ca is not None:
            for cutoff in [6, 8, 10]:
                count = 0

                for item in environment:
                    same_residue = (
                        item["chain_id"] == chain_id
                        and item["chain_index"] == index
                    )

                    if same_residue:
                        continue

                    if item["ca"] is None:
                        continue

                    distance = float(
                        np.linalg.norm(
                            target_ca - item["ca"]
                        )
                    )

                    if distance <= cutoff:
                        count += 1

                contact_counts[cutoff] = float(count)

        oxygen_distances = []

        if target_n is not None:
            for item in environment:
                if item["o"] is None:
                    continue

                same_chain_local = (
                    item["chain_id"] == chain_id
                    and abs(
                        item["chain_index"] - index
                    ) <= 1
                )

                if same_chain_local:
                    continue

                oxygen_distances.append(
                    float(
                        np.linalg.norm(
                            target_n - item["o"]
                        )
                    )
                )

        nearest_o_distance = (
            min(oxygen_distances)
            if oxygen_distances
            else np.nan
        )

        oxygen_count_3p5 = (
            float(
                sum(
                    distance <= 3.5
                    for distance in oxygen_distances
                )
            )
            if oxygen_distances
            else np.nan
        )

        hbond_mask = float(
            target_n is not None
            and len(oxygen_distances) > 0
        )

        residue_id = residue.id

        row = {
            "chain_id": chain_id,
            "pdb_seq_position": index + 1,
            "pdb_resseq": int(residue_id[1]),
            "pdb_icode": str(residue_id[2]).strip(),
            "comp_id": comp_id,
            "aa_code": aa_code,

            **one_hot_residue("aa_", aa_code),
            **one_hot_residue("prev_", previous_aa),
            **one_hot_residue("next_", next_aa),

            "position_fraction": float(
                (index + 1) / chain_length
            ),
            "log_chain_length": float(
                np.log(chain_length) / 6.0
            ),

            "sin_phi": (
                float(np.sin(phi))
                if np.isfinite(phi)
                else 0.0
            ),
            "cos_phi": (
                float(np.cos(phi))
                if np.isfinite(phi)
                else 0.0
            ),
            "phi_mask": phi_mask,

            "sin_psi": (
                float(np.sin(psi))
                if np.isfinite(psi)
                else 0.0
            ),
            "cos_psi": (
                float(np.cos(psi))
                if np.isfinite(psi)
                else 0.0
            ),
            "psi_mask": psi_mask,

            "ca_contacts_6a": contact_counts[6],
            "ca_contacts_8a": contact_counts[8],
            "ca_contacts_10a": contact_counts[10],
            "nearest_backbone_o_distance": nearest_o_distance,
            "backbone_o_within_3p5a": oxygen_count_3p5,
            "hbond_proxy_mask": hbond_mask,
        }

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


class NucleusSpecificHNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_trunk = nn.Sequential(
            nn.Linear(68, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        self.h_structure_branch = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
        )

        self.n_structure_branch = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
        )

        self.h_head = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

        self.n_head = nn.Sequential(
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        base_features,
        h_structure_features,
        n_structure_features,
    ):
        base_latent = self.base_trunk(base_features)

        h_latent = self.h_structure_branch(
            h_structure_features
        )
        n_latent = self.n_structure_branch(
            n_structure_features
        )

        pred_h = self.h_head(
            torch.cat(
                [base_latent, h_latent],
                dim=1,
            )
        )

        pred_n = self.n_head(
            torch.cat(
                [base_latent, n_latent],
                dim=1,
            )
        )

        return torch.cat(
            [pred_h, pred_n],
            dim=1,
        )


def preprocess_structure_columns(
    feature_frame,
    preprocessing_block,
):
    columns = list(preprocessing_block["columns"])

    values = (
        feature_frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .copy()
    )

    for column in columns:
        median = float(
            preprocessing_block["medians"][column]
        )
        mean = float(
            preprocessing_block["means"][column]
        )
        std = float(
            preprocessing_block["stds"][column]
        )

        if not np.isfinite(std) or std <= 0:
            std = 1.0

        values[column] = values[column].fillna(median)
        values[column] = (
            values[column] - mean
        ) / std

    return values.to_numpy(dtype=np.float32)


class V14Predictor:
    def __init__(self, release_dir, device=None):
        self.release_dir = Path(release_dir)

        preprocessing_path = (
            self.release_dir
            / "config"
            / "nucleus_specific_feature_preprocessing_v14.json"
        )

        feature_manifest_path = (
            self.release_dir
            / "config"
            / "feature_manifest.json"
        )

        if not preprocessing_path.exists():
            raise FileNotFoundError(preprocessing_path)

        if not feature_manifest_path.exists():
            raise FileNotFoundError(feature_manifest_path)

        with open(preprocessing_path) as handle:
            self.preprocessing = json.load(handle)

        with open(feature_manifest_path) as handle:
            self.feature_manifest = json.load(handle)

        self.base_columns = list(
            self.preprocessing["base_columns"]
        )
        self.h_block = self.preprocessing["h_structure"]
        self.n_block = self.preprocessing["n_structure"]
        self.target_scaling = self.preprocessing["target_scaling"]

        if (
            self.preprocessing.get("angle_feature_source")
            != "full_chain_pdb"
        ):
            raise RuntimeError(
                "Release does not specify full-chain PDB angles."
            )

        if (
            self.preprocessing.get("reference_correction")
            != "none"
        ):
            raise RuntimeError(
                "Unexpected reference correction configuration."
            )

        if self.preprocessing.get("model_contains_b_j"):
            raise RuntimeError(
                "Unexpected b_j configuration."
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        checkpoint_paths = sorted(
            (self.release_dir / "models").glob(
                "nucleus_specific_v14_seed_*.pt"
            )
        )

        if len(checkpoint_paths) != 5:
            raise RuntimeError(
                "Expected five v14 checkpoints, "
                f"found {len(checkpoint_paths)}."
            )

        self.models = []
        self.checkpoint_paths = checkpoint_paths

        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            if (
                checkpoint.get("angle_feature_source")
                != "full_chain_pdb"
            ):
                raise RuntimeError(
                    f"Invalid angle metadata: {checkpoint_path}"
                )

            if checkpoint.get(
                "reference_correction_applied",
                True,
            ):
                raise RuntimeError(
                    f"Unexpected reference correction: {checkpoint_path}"
                )

            if checkpoint.get(
                "hierarchical_b_j_used",
                True,
            ):
                raise RuntimeError(
                    f"Unexpected b_j metadata: {checkpoint_path}"
                )

            if (
                list(checkpoint["base_feature_columns"])
                != self.base_columns
            ):
                raise RuntimeError(
                    f"Base-feature mismatch: {checkpoint_path}"
                )

            model = NucleusSpecificHNModel().to(self.device)
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            model.eval()
            self.models.append(model)

    def predict_feature_frame(self, feature_frame):
        required_columns = (
            self.base_columns
            + list(self.h_block["columns"])
            + list(self.n_block["columns"])
        )

        missing_columns = sorted(
            set(required_columns)
            - set(feature_frame.columns)
        )

        if missing_columns:
            raise KeyError(
                f"Missing required feature columns: {missing_columns}"
            )

        base_array = (
            feature_frame[self.base_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )

        h_array = preprocess_structure_columns(
            feature_frame,
            self.h_block,
        )

        n_array = preprocess_structure_columns(
            feature_frame,
            self.n_block,
        )

        if not (
            np.isfinite(base_array).all()
            and np.isfinite(h_array).all()
            and np.isfinite(n_array).all()
        ):
            raise RuntimeError(
                "Non-finite values remain after preprocessing."
            )

        base_tensor = torch.from_numpy(
            base_array
        ).to(self.device)

        h_tensor = torch.from_numpy(
            h_array
        ).to(self.device)

        n_tensor = torch.from_numpy(
            n_array
        ).to(self.device)

        standardized_predictions = []

        with torch.inference_mode():
            for model in self.models:
                standardized_predictions.append(
                    model(
                        base_tensor,
                        h_tensor,
                        n_tensor,
                    ).cpu().numpy()
                )

        standardized_predictions = np.stack(
            standardized_predictions,
            axis=0,
        )

        ensemble_standardized = (
            standardized_predictions.mean(axis=0)
        )

        h_mean = float(
            self.target_scaling["shift_h_raw"]["mean"]
        )
        h_std = float(
            self.target_scaling["shift_h_raw"]["std"]
        )
        n_mean = float(
            self.target_scaling["shift_n_raw"]["mean"]
        )
        n_std = float(
            self.target_scaling["shift_n_raw"]["std"]
        )

        seed_h_ppm = (
            standardized_predictions[:, :, 0]
            * h_std
            + h_mean
        )

        seed_n_ppm = (
            standardized_predictions[:, :, 1]
            * n_std
            + n_mean
        )

        output = feature_frame[
            [
                "chain_id",
                "pdb_seq_position",
                "pdb_resseq",
                "pdb_icode",
                "comp_id",
                "aa_code",
            ]
        ].copy()

        output["pred_h_ppm"] = (
            ensemble_standardized[:, 0]
            * h_std
            + h_mean
        )

        output["pred_n_ppm"] = (
            ensemble_standardized[:, 1]
            * n_std
            + n_mean
        )

        output["ensemble_sd_h_ppm"] = (
            seed_h_ppm.std(axis=0, ddof=0)
        )

        output["ensemble_sd_n_ppm"] = (
            seed_n_ppm.std(axis=0, ddof=0)
        )

        output["reference_correction_applied"] = False
        output["hierarchical_b_j_used"] = False

        return output

    def predict_pdb(
        self,
        structure_path,
        chain_id,
        include_proline=False,
    ):
        features = build_pdb_feature_frame(
            structure_path,
            chain_id,
            include_proline=include_proline,
        )

        return self.predict_feature_frame(features)
