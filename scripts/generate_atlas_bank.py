#!/usr/bin/env python3
"""Build a compact prostate atlas/reference bank from PI-CAI ROI data.

Default input is the PI-CAI path_list.txt with rows:

    t2_path adc_path dwi_path prostate_zones_path lesion_mask_path

The path list currently stores older /raid/... paths. By default this script remaps
those paths to the local cluster dataset root before loading files.
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_CASE_LIST = (
    "/cluster/project7/longitude/Datasets/ProstateDatasets/3-picai-data/"
    "data-ROI-192-96/data_split_files/data_include_missing/path_list.txt"
)
DEFAULT_DATA_ROOT = (
    "/cluster/project7/longitude/Datasets/ProstateDatasets/3-picai-data/"
    "data-ROI-192-96"
)
LEGACY_DATA_ROOT = "/raid/candi/Wen/ProstateSeg/Data/scripts/3-picai-data/data-ROI-192-96"


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    return data, img.affine, img.header, spacing


def resolve_dataset_path(raw_path, data_root):
    if raw_path in (None, "", "None"):
        return None

    path = Path(raw_path)
    if path.exists():
        return str(path)

    text = str(path)
    if text.startswith(LEGACY_DATA_ROOT):
        remapped = Path(data_root) / Path(text).relative_to(LEGACY_DATA_ROOT)
        if remapped.exists():
            return str(remapped)

    parts = path.parts
    case_parts = [p for p in parts if p.startswith("P-")]
    if case_parts:
        candidate = Path(data_root) / case_parts[-1] / path.name
        if candidate.exists():
            return str(candidate)

    return str(path)


def read_path_list(path_list, data_root):
    rows = []
    with open(path_list) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 4:
                raise ValueError(f"{path_list}:{line_no}: expected at least 4 columns, got {len(cols)}")

            t2_path = resolve_dataset_path(cols[0], data_root)
            zones_path = resolve_dataset_path(cols[3], data_root)
            case_id = Path(t2_path).parent.name
            rows.append(
                {
                    "case_id": case_id,
                    "t2_path": t2_path,
                    "zones_path": zones_path,
                    "adc_path": resolve_dataset_path(cols[1], data_root) if len(cols) > 1 else None,
                    "dwi_path": resolve_dataset_path(cols[2], data_root) if len(cols) > 2 else None,
                    "lesion_path": resolve_dataset_path(cols[4], data_root) if len(cols) > 4 else None,
                }
            )
    return pd.DataFrame(rows)


def read_case_table(case_list, data_root):
    case_list = Path(case_list)
    if case_list.suffix.lower() == ".csv":
        cases = pd.read_csv(case_list)
        if "zones_path" not in cases.columns:
            required = {"case_id", "t2_path", "whole_mask_path", "pz_mask_path", "cg_mask_path"}
            if required.issubset(cases.columns):
                return cases
            raise ValueError(
                "CSV input must contain either zones_path or columns: "
                "case_id,t2_path,whole_mask_path,pz_mask_path,cg_mask_path"
            )
        for col in ["t2_path", "zones_path", "adc_path", "dwi_path", "lesion_path"]:
            if col in cases.columns:
                cases[col] = cases[col].map(lambda p: resolve_dataset_path(p, data_root))
        return cases
    return read_path_list(case_list, data_root)


def binarise(mask):
    return (mask > 0).astype(np.uint8)


def compute_volume_ml(mask, spacing):
    voxel_volume_mm3 = float(np.prod(spacing))
    volume_mm3 = np.sum(mask > 0) * voxel_volume_mm3
    return volume_mm3 / 1000.0


def get_bbox(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    size = maxs - mins + 1
    return {
        "x_min": int(mins[0]),
        "y_min": int(mins[1]),
        "z_min": int(mins[2]),
        "x_max": int(maxs[0]),
        "y_max": int(maxs[1]),
        "z_max": int(maxs[2]),
        "size_x_vox": int(size[0]),
        "size_y_vox": int(size[1]),
        "size_z_vox": int(size[2]),
    }


def compute_centroid(mask, spacing):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    return coords.mean(axis=0) * np.array(spacing)


def compute_shape_features(mask, spacing):
    bbox = get_bbox(mask)
    if bbox is None:
        raise ValueError("Empty whole prostate mask")

    size_x_mm = bbox["size_x_vox"] * spacing[0]
    size_y_mm = bbox["size_y_vox"] * spacing[1]
    size_z_mm = bbox["size_z_vox"] * spacing[2]
    dims = np.array([size_x_mm, size_y_mm, size_z_mm], dtype=float)
    sorted_dims = np.sort(dims)
    centroid = compute_centroid(mask, spacing)

    return {
        "bbox_x_mm": float(size_x_mm),
        "bbox_y_mm": float(size_y_mm),
        "bbox_z_mm": float(size_z_mm),
        "elongation": float(sorted_dims[-1] / max(sorted_dims[0], 1e-6)),
        "flatness": float(sorted_dims[0] / max(sorted_dims[-1], 1e-6)),
        "centroid_x_mm": float(centroid[0]),
        "centroid_y_mm": float(centroid[1]),
        "centroid_z_mm": float(centroid[2]),
    }


def safe_stats(values, prefix):
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def compute_intensity_features(t2, whole_mask, pz_mask, cg_mask):
    eps = 1e-6
    whole_values = t2[whole_mask > 0]
    pz_values = t2[pz_mask > 0]
    cg_values = t2[cg_mask > 0]

    features = {}
    features.update(safe_stats(whole_values, "whole_t2"))
    features.update(safe_stats(pz_values, "pz_t2"))
    features.update(safe_stats(cg_values, "cg_t2"))
    features["pz_cg_mean_ratio"] = (
        float(np.mean(pz_values) / (np.mean(cg_values) + eps))
        if pz_values.size > 0 and cg_values.size > 0
        else np.nan
    )
    return features


def masks_from_row(row, pz_label, cg_label):
    if "zones_path" in row and pd.notna(row["zones_path"]):
        zones, affine, header, spacing = load_nifti(row["zones_path"])
        pz = (zones == pz_label).astype(np.uint8)
        cg = (zones == cg_label).astype(np.uint8)
        whole = (zones > 0).astype(np.uint8)
        return whole, pz, cg, affine, header, spacing

    whole, affine, header, spacing = load_nifti(row["whole_mask_path"])
    pz, _, _, _ = load_nifti(row["pz_mask_path"])
    cg, _, _, _ = load_nifti(row["cg_mask_path"])
    return binarise(whole), binarise(pz), binarise(cg), affine, header, spacing


def extract_case_features(row, pz_label, cg_label):
    case_id = row["case_id"]
    t2, _, _, spacing = load_nifti(row["t2_path"])
    whole, pz, cg, _, _, mask_spacing = masks_from_row(row, pz_label, cg_label)

    if t2.shape != whole.shape:
        raise ValueError(f"{case_id}: T2 and zones shape mismatch: {t2.shape} vs {whole.shape}")
    if pz.sum() == 0 or cg.sum() == 0:
        raise ValueError(f"{case_id}: empty zone mask with pz_label={pz_label}, cg_label={cg_label}")

    whole_volume = compute_volume_ml(whole, spacing)
    pz_volume = compute_volume_ml(pz, spacing)
    cg_volume = compute_volume_ml(cg, spacing)
    zone_sum = pz_volume + cg_volume

    features = {
        "case_id": case_id,
        "t2_path": row["t2_path"],
        "zones_path": row.get("zones_path", ""),
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
        "whole_volume_ml": float(whole_volume),
        "pz_volume_ml": float(pz_volume),
        "cg_volume_ml": float(cg_volume),
        "pz_whole_ratio": float(pz_volume / whole_volume) if whole_volume > 0 else np.nan,
        "cg_whole_ratio": float(cg_volume / whole_volume) if whole_volume > 0 else np.nan,
        "pz_cg_ratio": float(pz_volume / cg_volume) if cg_volume > 0 else np.nan,
        "zone_coverage_ratio": float(zone_sum / whole_volume) if whole_volume > 0 else np.nan,
    }
    features.update(compute_shape_features(whole, spacing))
    features.update(compute_intensity_features(t2, whole, pz, cg))
    return features


def choose_feature_columns(df):
    candidate_cols = [
        "whole_volume_ml",
        "pz_volume_ml",
        "cg_volume_ml",
        "pz_whole_ratio",
        "cg_whole_ratio",
        "pz_cg_ratio",
        "bbox_x_mm",
        "bbox_y_mm",
        "bbox_z_mm",
        "elongation",
        "flatness",
        "whole_t2_mean",
        "whole_t2_std",
        "pz_t2_mean",
        "cg_t2_mean",
        "pz_cg_mean_ratio",
    ]
    return [c for c in candidate_cols if c in df.columns]


def cluster_cases(df, n_clusters):
    feature_cols = choose_feature_columns(df)
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = cluster_labels
    return df, scaler, kmeans, feature_cols, X_scaled


def select_representatives(df, X_scaled, kmeans, n_per_cluster):
    selected_indices = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_indices = np.where(df["cluster"].values == cluster_id)[0]
        centre = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(X_scaled[cluster_indices] - centre, axis=1)
        ranked_local_indices = np.argsort(distances)
        selected_indices.extend(cluster_indices[ranked_local_indices[:n_per_cluster]].tolist())

    selected_df = df.iloc[selected_indices].copy()
    selected_df["is_reference"] = True
    df = df.copy()
    df["is_reference"] = False
    df.loc[selected_indices, "is_reference"] = True
    return df, selected_df


def copy_reference_files(selected_df, output_dir, pz_label, cg_label):
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    copied_records = []
    for _, row in selected_df.iterrows():
        case_id = row["case_id"]
        t2_src = Path(row["t2_path"])
        zones_src = Path(row["zones_path"])
        t2_dst = image_dir / f"{case_id}_t2.nii.gz"
        zones_dst = mask_dir / f"{case_id}_prostate_zones.nii.gz"
        shutil.copy2(t2_src, t2_dst)
        shutil.copy2(zones_src, zones_dst)

        zones, affine, header, _ = load_nifti(zones_src)
        whole = (zones > 0).astype(np.uint8)
        pz = (zones == pz_label).astype(np.uint8)
        cg = (zones == cg_label).astype(np.uint8)
        nib.save(nib.Nifti1Image(whole, affine, header), mask_dir / f"{case_id}_whole.nii.gz")
        nib.save(nib.Nifti1Image(pz, affine, header), mask_dir / f"{case_id}_pz.nii.gz")
        nib.save(nib.Nifti1Image(cg, affine, header), mask_dir / f"{case_id}_cg.nii.gz")

        record = row.to_dict()
        record["bank_t2_path"] = str(t2_dst)
        record["bank_zones_path"] = str(zones_dst)
        copied_records.append(record)
    return pd.DataFrame(copied_records)


def save_metadata(df, selected_df, feature_cols, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_case_features.csv", index=False)
    selected_df.to_csv(output_dir / "reference_bank.csv", index=False)

    metadata = {
        "case_list": str(args.case_list),
        "data_root": str(args.data_root),
        "feature_columns": feature_cols,
        "n_total_cases": int(len(df)),
        "n_reference_cases": int(len(selected_df)),
        "clusters": sorted([int(x) for x in df["cluster"].unique()]),
        "n_clusters": int(args.n_clusters),
        "n_per_cluster": int(args.n_per_cluster),
        "pz_label": int(args.pz_label),
        "cg_label": int(args.cg_label),
        "copy_files": bool(args.copy_files),
    }
    with open(output_dir / "bank_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved all features to: {output_dir / 'all_case_features.csv'}")
    print(f"Saved reference bank to: {output_dir / 'reference_bank.csv'}")
    print(f"Saved metadata to: {output_dir / 'bank_metadata.json'}")


def main():
    parser = argparse.ArgumentParser(description="Generate prostate CG/PZ atlas reference bank")
    parser.add_argument("--case_list", default=DEFAULT_CASE_LIST, help="Path-list txt or CSV case table")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Dataset root used to remap stale path-list paths")
    parser.add_argument("--output_dir", default="register/atlas/atlas_bank", help="Output directory for bank metadata")
    parser.add_argument("--n_clusters", type=int, default=6, help="Number of anatomical clusters")
    parser.add_argument("--n_per_cluster", type=int, default=3, help="Reference cases selected from each cluster")
    parser.add_argument("--pz_label", type=int, default=2, help="Label value for peripheral zone in prostate_zones")
    parser.add_argument("--cg_label", type=int, default=1, help="Label value for central gland in prostate_zones")
    parser.add_argument("--copy_files", action="store_true", help="Copy selected images and derived masks into output_dir")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional debug limit on number of input cases")
    args = parser.parse_args()

    cases = read_case_table(args.case_list, args.data_root)
    if args.max_cases is not None:
        cases = cases.head(args.max_cases)
    print(f"Found {len(cases)} input cases from {args.case_list}")

    feature_rows = []
    failed = []
    for _, row in cases.iterrows():
        case_id = row["case_id"]
        print(f"Extracting features: {case_id}")
        try:
            feature_rows.append(extract_case_features(row, args.pz_label, args.cg_label))
        except Exception as exc:
            failed.append({"case_id": case_id, "error": str(exc)})
            print(f"Failed case {case_id}: {exc}")

    feature_df = pd.DataFrame(feature_rows)
    if len(feature_df) == 0:
        raise RuntimeError("No valid cases found")

    clustered_df, scaler, kmeans, feature_cols, X_scaled = cluster_cases(feature_df, args.n_clusters)
    clustered_df, selected_df = select_representatives(clustered_df, X_scaled, kmeans, args.n_per_cluster)

    output_dir = Path(args.output_dir)
    if args.copy_files:
        selected_df = copy_reference_files(selected_df, output_dir, args.pz_label, args.cg_label)

    save_metadata(clustered_df, selected_df, feature_cols, output_dir, args)
    if failed:
        failed_path = output_dir / "failed_cases.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Saved failed case report to: {failed_path}")

    print("\nSelected reference cases:")
    print(selected_df[["case_id", "cluster", "whole_volume_ml", "pz_whole_ratio", "cg_whole_ratio"]])


if __name__ == "__main__":
    main()
