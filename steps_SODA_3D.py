from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from skimage import exposure, measure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ome_3d import OmeTiffInfo, load_ome_tiff_channel_zyx


@dataclass
class SpotTable:
    name: str
    channel: int
    spots: pd.DataFrame
    labeled: np.ndarray


def _torch_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _empty_torch_cache(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def create_3d_log_kernel(sigma: tuple[float, float, float], device, truncate: float = 4.0):
    import torch

    sigma_z, sigma_y, sigma_x = sigma
    radius_z = math.ceil(truncate * sigma_z)
    radius_y = math.ceil(truncate * sigma_y)
    radius_x = math.ceil(truncate * sigma_x)

    z = torch.arange(2 * radius_z + 1, dtype=torch.float32, device=device) - radius_z
    y = torch.arange(2 * radius_y + 1, dtype=torch.float32, device=device) - radius_y
    x = torch.arange(2 * radius_x + 1, dtype=torch.float32, device=device) - radius_x
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    gaussian = torch.exp(
        -(
            zz**2 / (2 * sigma_z**2)
            + yy**2 / (2 * sigma_y**2)
            + xx**2 / (2 * sigma_x**2)
        )
    )
    laplacian = (
        zz**2 / sigma_z**4
        + yy**2 / sigma_y**4
        + xx**2 / sigma_x**4
        - (1 / sigma_z**2 + 1 / sigma_y**2 + 1 / sigma_x**2)
    ) * gaussian
    return laplacian.unsqueeze(0).unsqueeze(0)


def create_1d_log_kernels(sigma: tuple[float, float, float], device, truncate: float = 4.0):
    import torch

    kernels = []
    for axis_sigma in sigma:
        radius = math.ceil(truncate * axis_sigma)
        coords = torch.arange(2 * radius + 1, dtype=torch.float32, device=device) - radius
        gaussian = torch.exp(-(coords**2) / (2 * axis_sigma**2))
        second = ((coords**2 / axis_sigma**4) - (1 / axis_sigma**2)) * gaussian
        kernels.append((gaussian, second))
    return kernels


def _conv1d_axis_3d(tensor, kernel_1d, axis: str):
    import torch.nn.functional as F

    radius = int(kernel_1d.numel() // 2)
    if axis == "z":
        weight = kernel_1d.reshape(1, 1, -1, 1, 1)
        padding = (0, 0, 0, 0, radius, radius)
    elif axis == "y":
        weight = kernel_1d.reshape(1, 1, 1, -1, 1)
        padding = (0, 0, radius, radius, 0, 0)
    elif axis == "x":
        weight = kernel_1d.reshape(1, 1, 1, 1, -1)
        padding = (radius, radius, 0, 0, 0, 0)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return F.conv3d(F.pad(tensor, padding, mode="reflect"), weight)


def separable_log_response_3d(tensor, sigma: tuple[float, float, float], device, truncate: float = 4.0):
    """Compute unnormalised 3D LoG response with separable 1D convolutions."""
    (gz, gzz), (gy, gyy), (gx, gxx) = create_1d_log_kernels(sigma, device, truncate=truncate)

    z_term = _conv1d_axis_3d(tensor, gzz, "z")
    z_term = _conv1d_axis_3d(z_term, gy, "y")
    z_term = _conv1d_axis_3d(z_term, gx, "x")

    y_term = _conv1d_axis_3d(tensor, gz, "z")
    y_term = _conv1d_axis_3d(y_term, gyy, "y")
    y_term = _conv1d_axis_3d(y_term, gx, "x")

    x_term = _conv1d_axis_3d(tensor, gz, "z")
    x_term = _conv1d_axis_3d(x_term, gy, "y")
    x_term = _conv1d_axis_3d(x_term, gxx, "x")

    return -(z_term + y_term + x_term)


def separable_log_response_3d_tiled(
    volume: np.ndarray,
    sigma: tuple[float, float, float],
    device,
    truncate: float = 4.0,
    tile_yx: int = 224,
) -> np.ndarray:
    """Compute separable LoG using YX tiles.

    PyTorch MPS currently gives incorrect 1D conv3d results for some larger
    XY sizes (notably 256x256). Tiling with halos keeps tensors in the reliable
    size range while preserving the same result in each tile interior.
    """
    import torch

    z_size, y_size, x_size = volume.shape
    radius_y = math.ceil(truncate * sigma[1])
    radius_x = math.ceil(truncate * sigma[2])
    response = np.empty(volume.shape, dtype=np.float32)

    for y0 in range(0, y_size, tile_yx):
        y1 = min(y0 + tile_yx, y_size)
        yy0 = max(0, y0 - radius_y)
        yy1 = min(y_size, y1 + radius_y)
        crop_y0 = y0 - yy0
        crop_y1 = crop_y0 + (y1 - y0)

        for x0 in range(0, x_size, tile_yx):
            x1 = min(x0 + tile_yx, x_size)
            xx0 = max(0, x0 - radius_x)
            xx1 = min(x_size, x1 + radius_x)
            crop_x0 = x0 - xx0
            crop_x1 = crop_x0 + (x1 - x0)

            tile = np.ascontiguousarray(volume[:, yy0:yy1, xx0:xx1])
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                tile_response = separable_log_response_3d(
                    tile_tensor,
                    sigma,
                    device,
                    truncate=truncate,
                )
            response[:, y0:y1, x0:x1] = (
                tile_response[0, 0, :, crop_y0:crop_y1, crop_x0:crop_x1]
                .detach()
                .cpu()
                .numpy()
            )
            del tile_tensor, tile_response
    return response


def detect_3d_log_spots(
    volume: np.ndarray,
    mask: np.ndarray,
    sigma: tuple[float, float, float] = (1.0, 2.0, 2.0),
    threshold_multiplier: float = 8.0,
    min_volume: int = 30,
    chunk_size: int = 16,
    return_labeled: bool = False,
    require_mps: bool = False,
    log_method: str = "separable",
    truncate: float = 4.0,
    tile_yx: int = 224,
) -> tuple[list[measure._regionprops.RegionProperties], np.ndarray | None, float]:
    import torch
    import torch.nn.functional as F

    if volume.shape != mask.shape:
        raise ValueError(f"Volume shape {volume.shape} does not match mask shape {mask.shape}.")

    device = _torch_device()
    if require_mps and device.type != "mps":
        raise RuntimeError(
            "MPS was required, but PyTorch did not report an available MPS device."
        )
    if log_method not in {"dense", "separable"}:
        raise ValueError("log_method must be 'dense' or 'separable'.")
    print(f"3D LoG detection using torch device: {device}, method: {log_method}, truncate: {truncate}")
    kernel = None
    padding = None
    if log_method == "dense":
        kernel = create_3d_log_kernel(sigma, device, truncate=truncate)
        kernel_shape = tuple(int(v) for v in kernel.shape[2:])
        padding = (
            kernel_shape[2] // 2,
            kernel_shape[2] // 2,
            kernel_shape[1] // 2,
            kernel_shape[1] // 2,
            kernel_shape[0] // 2,
            kernel_shape[0] // 2,
        )
    else:
        kernel_shape = tuple(2 * math.ceil(truncate * axis_sigma) + 1 for axis_sigma in sigma)

    sample_step = max(1, volume.shape[0] // 10)
    sample = volume[::sample_step].astype(np.float32) / 65535.0
    sample_mask = mask[::sample_step]
    with torch.no_grad():
        if log_method == "dense":
            sample_tensor = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0).to(device)
            sample_response = -F.conv3d(F.pad(sample_tensor, padding, mode="reflect"), kernel)
            del sample_tensor
        else:
            sample_response = separable_log_response_3d_tiled(
                sample,
                sigma,
                device,
                truncate=truncate,
                tile_yx=tile_yx,
            )
    if log_method == "dense":
        sample_response_np = sample_response[0, 0].detach().cpu().numpy()
    else:
        sample_response_np = sample_response
    if np.any(sample_mask):
        response_values = sample_response_np[sample_mask]
    else:
        response_values = sample_response_np.ravel()
    threshold = float(response_values.mean() + threshold_multiplier * response_values.std())
    del sample, sample_response, sample_response_np
    _empty_torch_cache(device)

    z_size = volume.shape[0]
    overlap = kernel_shape[0] // 2
    binary = np.zeros(volume.shape, dtype=bool)

    for z_start in range(0, z_size, chunk_size):
        z_end = min(z_start + chunk_size, z_size)
        chunk_start = max(0, z_start - overlap)
        chunk_end = min(z_size, z_end + overlap)
        chunk = volume[chunk_start:chunk_end].astype(np.float32) / 65535.0
        if log_method == "dense":
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                response = -F.conv3d(F.pad(chunk_tensor, padding, mode="reflect"), kernel)
            response_np = response[0, 0].detach().cpu().numpy()
            del chunk_tensor, response
        else:
            response_np = separable_log_response_3d_tiled(
                chunk,
                sigma,
                device,
                truncate=truncate,
                tile_yx=tile_yx,
            )
        crop_start = z_start - chunk_start
        crop_end = crop_start + (z_end - z_start)
        binary[z_start:z_end] = response_np[crop_start:crop_end] > threshold
        del chunk, response_np
        _empty_torch_cache(device)

    binary &= mask
    labeled = measure.label(binary, connectivity=1)
    props = measure.regionprops(labeled, intensity_image=volume)
    filtered = [spot for spot in props if spot.area >= min_volume]

    filtered_labeled = None
    if return_labeled:
        mapping = np.zeros(labeled.max() + 1, dtype=np.uint32)
        for new_label, spot in enumerate(filtered, start=1):
            mapping[spot.label] = new_label
        filtered_labeled = mapping[labeled]

    return filtered, filtered_labeled, threshold


def _spots_to_dataframe(
    spots: list[measure._regionprops.RegionProperties],
    channel_name: str,
    channel_index: int,
    voxel_size_zyx_um: tuple[float, float, float],
) -> pd.DataFrame:
    rows = []
    z_um, y_um, x_um = voxel_size_zyx_um
    for i, spot in enumerate(spots, start=1):
        try:
            centroid_z, centroid_y, centroid_x = spot.weighted_centroid
            if not np.all(np.isfinite([centroid_z, centroid_y, centroid_x])):
                raise ValueError
        except Exception:
            centroid_z, centroid_y, centroid_x = spot.centroid
        z0, y0, x0, z1, y1, x1 = spot.bbox
        rows.append(
            {
                "spot_id": i,
                "channel_name": channel_name,
                "channel_index": channel_index,
                "centroid_z": float(centroid_z),
                "centroid_y": float(centroid_y),
                "centroid_x": float(centroid_x),
                "centroid_z_um": float(centroid_z * z_um),
                "centroid_y_um": float(centroid_y * y_um),
                "centroid_x_um": float(centroid_x * x_um),
                "volume_voxels": int(spot.area),
                "mean_intensity": float(spot.intensity_mean),
                "max_intensity": float(spot.intensity_max),
                "bbox_z0": int(z0),
                "bbox_y0": int(y0),
                "bbox_x0": int(x0),
                "bbox_z1": int(z1),
                "bbox_y1": int(y1),
                "bbox_x1": int(x1),
            }
        )
    return pd.DataFrame(rows)


def _physical_coords(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 3), dtype=np.float64)
    return df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(dtype=np.float64)


def nearest_neighbor_table(focus_df: pd.DataFrame, subject_df: pd.DataFrame) -> pd.DataFrame:
    focus_coords = _physical_coords(focus_df)
    subject_coords = _physical_coords(subject_df)
    if len(focus_coords) == 0 or len(subject_coords) == 0:
        return pd.DataFrame()

    subject_tree = cKDTree(subject_coords)
    distances, indices = subject_tree.query(focus_coords, k=1)
    rows = []
    for focus_row, distance, subject_index in zip(focus_df.itertuples(index=False), distances, indices):
        subject_row = subject_df.iloc[int(subject_index)]
        rows.append(
            {
                "focus_spot_id": int(focus_row.spot_id),
                "subject_spot_id": int(subject_row.spot_id),
                "distance_um": float(distance),
                "focus_z_um": float(focus_row.centroid_z_um),
                "focus_y_um": float(focus_row.centroid_y_um),
                "focus_x_um": float(focus_row.centroid_x_um),
                "subject_z_um": float(subject_row.centroid_z_um),
                "subject_y_um": float(subject_row.centroid_y_um),
                "subject_x_um": float(subject_row.centroid_x_um),
            }
        )
    return pd.DataFrame(rows)


def _sample_random_points_in_mask(
    mask: np.ndarray,
    n_points: int,
    voxel_size_zyx_um: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0 or n_points == 0:
        return np.empty((0, 3), dtype=np.float64)
    selected = coords[rng.integers(0, len(coords), size=n_points)]
    scale = np.array(voxel_size_zyx_um, dtype=np.float64)
    return selected.astype(np.float64) * scale


def shell_statistics(
    focus_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    mask: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float],
    shell_width_um: float = 0.05,
    n_shells: int = 20,
    random_iterations: int = 100,
    random_seed: int = 7,
) -> pd.DataFrame:
    focus_coords = _physical_coords(focus_df)
    subject_coords = _physical_coords(subject_df)
    edges = np.arange(n_shells + 1, dtype=np.float64) * shell_width_um
    max_distance = float(edges[-1])
    rows = []

    if len(focus_coords) == 0 or len(subject_coords) == 0:
        for i in range(n_shells):
            rows.append(
                {
                    "shell_index": i,
                    "inner_radius_um": float(edges[i]),
                    "outer_radius_um": float(edges[i + 1]),
                    "focus_with_subject_probability": np.nan,
                    "mean_subject_count_per_focus": np.nan,
                    "random_focus_with_subject_probability": np.nan,
                    "random_mean_subject_count_per_focus": np.nan,
                    "probability_enrichment": np.nan,
                    "count_enrichment": np.nan,
                }
            )
        return pd.DataFrame(rows)

    subject_tree = cKDTree(subject_coords)
    cumulative_neighbors = subject_tree.query_ball_point(focus_coords, r=max_distance)
    observed_counts = np.zeros((len(focus_coords), n_shells), dtype=np.int32)
    for focus_idx, neighbors in enumerate(cumulative_neighbors):
        if not neighbors:
            continue
        distances = np.linalg.norm(subject_coords[neighbors] - focus_coords[focus_idx], axis=1)
        shell_ids = np.searchsorted(edges, distances, side="right") - 1
        valid = (shell_ids >= 0) & (shell_ids < n_shells)
        if np.any(valid):
            observed_counts[focus_idx] = np.bincount(shell_ids[valid], minlength=n_shells)[:n_shells]

    rng = np.random.default_rng(random_seed)
    random_count_acc = np.zeros(n_shells, dtype=np.float64)
    random_prob_acc = np.zeros(n_shells, dtype=np.float64)
    iterations_done = 0
    for _ in range(random_iterations):
        random_subject = _sample_random_points_in_mask(mask, len(subject_coords), voxel_size_zyx_um, rng)
        if len(random_subject) == 0:
            continue
        random_tree = cKDTree(random_subject)
        random_neighbors = random_tree.query_ball_point(focus_coords, r=max_distance)
        random_counts = np.zeros((len(focus_coords), n_shells), dtype=np.int32)
        for focus_idx, neighbors in enumerate(random_neighbors):
            if not neighbors:
                continue
            distances = np.linalg.norm(random_subject[neighbors] - focus_coords[focus_idx], axis=1)
            shell_ids = np.searchsorted(edges, distances, side="right") - 1
            valid = (shell_ids >= 0) & (shell_ids < n_shells)
            if np.any(valid):
                random_counts[focus_idx] = np.bincount(shell_ids[valid], minlength=n_shells)[:n_shells]
        random_count_acc += random_counts.mean(axis=0)
        random_prob_acc += (random_counts > 0).mean(axis=0)
        iterations_done += 1

    random_count = random_count_acc / iterations_done if iterations_done else np.full(n_shells, np.nan)
    random_prob = random_prob_acc / iterations_done if iterations_done else np.full(n_shells, np.nan)
    observed_count = observed_counts.mean(axis=0)
    observed_prob = (observed_counts > 0).mean(axis=0)

    for i in range(n_shells):
        rows.append(
            {
                "shell_index": i,
                "inner_radius_um": float(edges[i]),
                "outer_radius_um": float(edges[i + 1]),
                "focus_with_subject_probability": float(observed_prob[i]),
                "mean_subject_count_per_focus": float(observed_count[i]),
                "random_focus_with_subject_probability": float(random_prob[i]),
                "random_mean_subject_count_per_focus": float(random_count[i]),
                "probability_enrichment": float(observed_prob[i] - random_prob[i]),
                "count_enrichment": float(observed_count[i] - random_count[i]),
            }
        )
    return pd.DataFrame(rows)


def _load_mask(mask_path: str | Path, z_slices: int, crop: tuple[slice, slice, slice] | None) -> np.ndarray:
    mask_2d = tifffile.imread(mask_path)
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask, got {mask_2d.shape}.")
    mask_2d = mask_2d > 0
    mask = np.broadcast_to(mask_2d, (z_slices, *mask_2d.shape))
    if crop is not None:
        z_crop, y_crop, x_crop = crop
        mask = mask[z_crop, y_crop, x_crop]
    return np.asarray(mask, dtype=bool)


def _write_qc_plot(
    output_path: Path,
    focus_volume: np.ndarray,
    subject_volume: np.ndarray,
    focus_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    mask: np.ndarray,
    focus_name: str,
    subject_name: str,
) -> None:
    focus_max = np.max(focus_volume.astype(np.float32), axis=0)
    subject_max = np.max(subject_volume.astype(np.float32), axis=0)
    mask_max = np.max(mask, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, image, df, title, color in [
        (axes[0], focus_max, focus_df, focus_name, "cyan"),
        (axes[1], subject_max, subject_df, subject_name, "magenta"),
    ]:
        ax.imshow(exposure.equalize_adapthist(image / max(float(image.max()), 1.0)), cmap="gray")
        ax.contour(mask_max, levels=[0.5], colors="yellow", linewidths=0.4, alpha=0.6)
        if not df.empty:
            ax.scatter(df["centroid_x"], df["centroid_y"], s=8, c=color, edgecolors="none", alpha=0.55)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def run_3d_soda(
    image_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    focus_channel: int = 0,
    subject_channel: int = 2,
    focus_name: str = "CTD",
    subject_name: str = "MED1",
    log_sigma_zyx: tuple[float, float, float] = (1.0, 2.0, 2.0),
    threshold_multiplier: float = 8.0,
    min_volume: int = 30,
    shell_width_um: float = 0.05,
    n_shells: int = 20,
    random_iterations: int = 100,
    chunk_size: int = 16,
    crop: tuple[slice, slice, slice] | None = None,
    random_seed: int = 7,
    save_label_images: bool = False,
    require_mps: bool = False,
    log_method: str = "separable",
    truncate: float = 4.0,
    tile_yx: int = 224,
) -> dict[str, object]:
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    focus_volume, info = load_ome_tiff_channel_zyx(image_path, focus_channel, crop=crop)
    subject_volume, _ = load_ome_tiff_channel_zyx(image_path, subject_channel, crop=crop)
    if focus_volume.shape != subject_volume.shape:
        raise ValueError(f"Channel shapes differ: {focus_volume.shape} vs {subject_volume.shape}.")

    mask = _load_mask(mask_path, info.size_z, crop=crop)
    if mask.shape != focus_volume.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {focus_volume.shape}.")

    voxel_size = (
        info.physical_size_z_um,
        info.physical_size_y_um,
        info.physical_size_x_um,
    )

    focus_spots, focus_labeled, focus_threshold = detect_3d_log_spots(
        focus_volume,
        mask,
        sigma=log_sigma_zyx,
        threshold_multiplier=threshold_multiplier,
        min_volume=min_volume,
        chunk_size=chunk_size,
        return_labeled=save_label_images,
        require_mps=require_mps,
        log_method=log_method,
        truncate=truncate,
        tile_yx=tile_yx,
    )
    subject_spots, subject_labeled, subject_threshold = detect_3d_log_spots(
        subject_volume,
        mask,
        sigma=log_sigma_zyx,
        threshold_multiplier=threshold_multiplier,
        min_volume=min_volume,
        chunk_size=chunk_size,
        return_labeled=save_label_images,
        require_mps=require_mps,
        log_method=log_method,
        truncate=truncate,
        tile_yx=tile_yx,
    )

    focus_df = _spots_to_dataframe(focus_spots, focus_name, focus_channel, voxel_size)
    subject_df = _spots_to_dataframe(subject_spots, subject_name, subject_channel, voxel_size)
    nn_df = nearest_neighbor_table(focus_df, subject_df)
    shell_df = shell_statistics(
        focus_df,
        subject_df,
        mask,
        voxel_size,
        shell_width_um=shell_width_um,
        n_shells=n_shells,
        random_iterations=random_iterations,
        random_seed=random_seed,
    )

    max_distance = shell_width_um * n_shells
    coupling_index_focus = (
        float((nn_df["distance_um"] <= max_distance).mean())
        if not nn_df.empty else np.nan
    )
    subject_nn_df = nearest_neighbor_table(subject_df, focus_df)
    coupling_index_subject = (
        float((subject_nn_df["distance_um"] <= max_distance).mean())
        if not subject_nn_df.empty else np.nan
    )
    coupled_distances = nn_df.loc[nn_df["distance_um"] <= max_distance, "distance_um"] if not nn_df.empty else pd.Series(dtype=float)

    summary = {
        "image": str(image_path),
        "mask": str(mask_path),
        "output_dir": str(output_dir),
        "axes": "CZYX",
        "shape_zyx": list(focus_volume.shape),
        "focus_channel": focus_channel,
        "subject_channel": subject_channel,
        "focus_name": focus_name,
        "subject_name": subject_name,
        "n_focus_spots": int(len(focus_df)),
        "n_subject_spots": int(len(subject_df)),
        "focus_threshold": focus_threshold,
        "subject_threshold": subject_threshold,
        "xy_um_per_pixel": info.physical_size_x_um,
        "z_um_per_slice": info.physical_size_z_um,
        "shell_width_um": shell_width_um,
        "n_shells": n_shells,
        "log_method": log_method,
        "truncate": truncate,
        "tile_yx": tile_yx,
        "max_shell_distance_um": max_distance,
        "coupling_index_focus_to_subject": coupling_index_focus,
        "coupling_index_subject_to_focus": coupling_index_subject,
        "mean_coupling_distance_um": float(coupled_distances.mean()) if len(coupled_distances) else np.nan,
        "crop": str(crop),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    sample = image_path.name.replace(".ome.tiff", "").replace(".ome.tif", "")
    focus_df.to_csv(output_dir / f"{sample}_{focus_name}_spots_3d.csv", index=False)
    subject_df.to_csv(output_dir / f"{sample}_{subject_name}_spots_3d.csv", index=False)
    nn_df.to_csv(output_dir / f"{sample}_{focus_name}_{subject_name}_nearest_neighbors_3d.csv", index=False)
    shell_df.to_csv(output_dir / f"{sample}_{focus_name}_{subject_name}_shell_probabilities_3d.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / f"{sample}_summary_3d.csv", index=False)

    with pd.ExcelWriter(output_dir / f"{sample}_pySODA_3d_results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        focus_df.to_excel(writer, sheet_name=f"{focus_name}_spots", index=False)
        subject_df.to_excel(writer, sheet_name=f"{subject_name}_spots", index=False)
        nn_df.to_excel(writer, sheet_name="nearest_neighbors", index=False)
        shell_df.to_excel(writer, sheet_name="shell_probabilities", index=False)

    if save_label_images:
        tifffile.imwrite(output_dir / f"{sample}_{focus_name}_spot_labels_3d.tif", focus_labeled, metadata={"axes": "ZYX"})
        tifffile.imwrite(output_dir / f"{sample}_{subject_name}_spot_labels_3d.tif", subject_labeled, metadata={"axes": "ZYX"})
    _write_qc_plot(
        output_dir / f"{sample}_QC_max_projection_spots_3d.png",
        focus_volume,
        subject_volume,
        focus_df,
        subject_df,
        mask,
        focus_name,
        subject_name,
    )

    with open(output_dir / f"{sample}_parameters_3d.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary
