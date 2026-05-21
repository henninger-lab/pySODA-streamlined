from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

import steps_SODA
from czyx_loader import load_image_cyx_max


def _input_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    files = sorted(path for path in globber(pattern) if path.is_file())
    return [
        path for path in files
        if path.suffix.lower() in {".tif", ".tiff", ".czi"}
        or path.name.lower().endswith((".ome.tif", ".ome.tiff"))
    ]


def _prepared_name(path: Path, channels: list[int]) -> str:
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    channel_label = "-".join(str(ch) for ch in channels)
    return f"{name}_channels_{channel_label}_max2d.tif"


def _expand_values(values: list[str], cast, n_channels: int, name: str) -> list:
    if len(values) == 1:
        return [cast(values[0])] * n_channels
    if len(values) != n_channels:
        raise argparse.ArgumentTypeError(
            f"{name} expects either 1 value or {n_channels} values matching --channels."
        )
    return [cast(value) for value in values]


def _expand_scale_list(values: list[str], n_channels: int) -> list[list[int]]:
    if len(values) == 1:
        values = values * n_channels
    if len(values) != n_channels:
        raise argparse.ArgumentTypeError(
            f"--scale-list expects either 1 value or {n_channels} values matching --channels."
        )
    return [[int(scale) for scale in channel.split(",") if scale] for channel in values]


def prepare_inputs(
    input_dir: Path,
    prepared_dir: Path,
    channels: list[int],
    pattern: str,
    recursive: bool,
    time_index: int,
) -> list[Path]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_files = []
    for source in _input_files(input_dir, pattern, recursive):
        out_path = prepared_dir / _prepared_name(source, channels)
        selected = load_image_cyx_max(source, channels=channels, time_index=time_index)
        if selected.ndim != 3:
            raise ValueError(f"Prepared image must be CYX, got {selected.shape} for {source}.")
        tifffile.imwrite(
            out_path,
            selected,
            compression="zstd",
            compressionargs={"level": 6},
            metadata={"axes": "CYX", "source_channels": channels, "source_file": str(source)},
        )
        prepared_files.append(out_path)
        print(f"Prepared {source.name} -> {out_path.name} {selected.shape}")
    return prepared_files


def write_combined_tables(output_dir: Path, prepared_dir: Path) -> None:
    results_path = output_dir / f"pySODA_results_{prepared_dir.name}.xlsx"
    if not results_path.exists():
        print(f"Combined workbook not found: {results_path}")
        return

    sheets = pd.read_excel(results_path, sheet_name=None)
    rows = []
    for channel_pair, frame in sheets.items():
        frame = frame.copy()
        frame.insert(0, "Channel pair", channel_pair)
        rows.append(frame)
    if not rows:
        return

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "combined_pySODA_results.csv", index=False)

    numeric_cols = combined.select_dtypes(include=["number"]).columns
    grouped = combined.groupby("Channel pair")[numeric_cols].agg(["count", "mean", "std"])
    sem = combined.groupby("Channel pair")[numeric_cols].sem(numeric_only=True)
    pooled = _pooled_pair_stats(combined)
    rings, pooled_rings = _combined_ring_tables(output_dir, combined)

    pooled.to_csv(output_dir / "pooled_pySODA_stats.csv", index=False)
    if rings is not None:
        rings.to_csv(output_dir / "combined_SODA_rings.csv", index=False)
    if pooled_rings is not None:
        pooled_rings.to_csv(output_dir / "pooled_SODA_rings.csv", index=False)
        _write_pooled_ring_plots(output_dir, pooled_rings)

    with pd.ExcelWriter(output_dir / "combined_pySODA_results.xlsx", engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="per_file", index=False)
        pooled.to_excel(writer, sheet_name="pooled_stats", index=False)
        grouped.to_excel(writer, sheet_name="summary_stats")
        sem.to_excel(writer, sheet_name="sem")
        if rings is not None:
            rings.to_excel(writer, sheet_name="per_file_rings", index=False)
        if pooled_rings is not None:
            pooled_rings.to_excel(writer, sheet_name="pooled_rings", index=False)


def _sem(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return float("nan")
    return float(series.std(ddof=1) / np.sqrt(len(series)))


def _pooled_pair_stats(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel_pair, frame in combined.groupby("Channel pair"):
        spots0 = pd.to_numeric(frame["Spots in channel 0"], errors="coerce")
        spots1 = pd.to_numeric(frame["Spots in channel 1"], errors="coerce")
        coupling0 = pd.to_numeric(frame["Coupling index 0"], errors="coerce")
        mean_distance = pd.to_numeric(frame["Weighted mean coupling distance"], errors="coerce")
        probability_mass = coupling0 * spots0
        total_probability_mass = float(probability_mass.sum())
        total_spots0 = float(spots0.sum())
        total_spots1 = float(spots1.sum())

        if total_probability_mass > 0:
            pooled_mean_distance = float((mean_distance * probability_mass).sum() / total_probability_mass)
        else:
            pooled_mean_distance = float("nan")

        rows.append({
            "Channel pair": channel_pair,
            "Files": int(len(frame)),
            "Total spots channel 0": int(total_spots0),
            "Total spots channel 1": int(total_spots1),
            "Total couples": int(pd.to_numeric(frame["Number of couples"], errors="coerce").sum()),
            "Total coupling probability mass": total_probability_mass,
            "Pooled coupling index 0": total_probability_mass / total_spots0 if total_spots0 else np.nan,
            "Pooled coupling index 1": total_probability_mass / total_spots1 if total_spots1 else np.nan,
            "Pooled coupling percent 0": 100 * total_probability_mass / total_spots0 if total_spots0 else np.nan,
            "Pooled coupling percent 1": 100 * total_probability_mass / total_spots1 if total_spots1 else np.nan,
            "Pooled weighted mean coupling distance": pooled_mean_distance,
            "Mean per-file coupling percent 0": float(frame["Coupling percent 0"].mean()),
            "SEM per-file coupling percent 0": _sem(frame["Coupling percent 0"]),
            "Mean per-file coupling percent 1": float(frame["Coupling percent 1"].mean()),
            "SEM per-file coupling percent 1": _sem(frame["Coupling percent 1"]),
            "Mean per-file weighted mean coupling distance": float(mean_distance.mean()),
            "SEM per-file weighted mean coupling distance": _sem(mean_distance),
            "Median SODA log10(p-value)": float(pd.to_numeric(frame["SODA log10(p-value)"], errors="coerce").median()),
            "Files with significant rings": int((pd.to_numeric(frame["Number of significant rings"], errors="coerce") > 0).sum()),
        })
    return pd.DataFrame(rows)


def _channel_pair_suffix(channel_pair: str) -> str:
    parts = channel_pair.replace("ch", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected channel pair label: {channel_pair}")
    return f"ch{parts[0]}{parts[1]}"


def _combined_ring_tables(output_dir: Path, combined: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    ring_rows = []
    for _, row in combined.iterrows():
        channel_pair = row["Channel pair"]
        file_name = row["File"]
        workbook = output_dir / f"pySODA_{file_name}_{_channel_pair_suffix(channel_pair)}.xlsx"
        if not workbook.exists():
            continue
        try:
            rings = pd.read_excel(workbook, sheet_name="SODA rings")
        except ValueError:
            continue
        rings.insert(0, "File", file_name)
        rings.insert(0, "Channel pair", channel_pair)
        ring_rows.append(rings)

    if not ring_rows:
        return None, None

    rings = pd.concat(ring_rows, ignore_index=True)
    grouped = rings.groupby(["Channel pair", "Ring index"], as_index=False)
    pooled = grouped.agg(
        files=("File", "count"),
        inner_radius_px=("Inner radius (px)", "mean"),
        outer_radius_px=("Outer radius (px)", "mean"),
        mean_coupling_probability=("Coupling probability", "mean"),
        std_coupling_probability=("Coupling probability", "std"),
        mean_G0=("G0", "mean"),
        std_G0=("G0", "std"),
        fraction_significant=("Significant", "mean"),
    )
    sem_cp = (
        rings.groupby(["Channel pair", "Ring index"])["Coupling probability"]
        .sem()
        .reset_index(name="sem_coupling_probability")
    )
    sem_g0 = (
        rings.groupby(["Channel pair", "Ring index"])["G0"]
        .sem()
        .reset_index(name="sem_G0")
    )
    pooled = pooled.merge(sem_cp, on=["Channel pair", "Ring index"], how="left")
    pooled = pooled.merge(sem_g0, on=["Channel pair", "Ring index"], how="left")
    return rings, pooled


def _write_pooled_ring_plots(output_dir: Path, pooled_rings: pd.DataFrame) -> None:
    for channel_pair, frame in pooled_rings.groupby("Channel pair"):
        frame = frame.sort_values("Ring index")
        x = frame["inner_radius_px"]
        y = frame["mean_coupling_probability"]
        yerr = frame["sem_coupling_probability"].fillna(0)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=1.5, capsize=3)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Distance ring start (px)")
        ax.set_ylabel("Mean coupling probability")
        ax.set_title(f"Pooled SODA rings: {channel_pair}")
        fig.tight_layout()
        fig.savefig(output_dir / f"pooled_SODA_rings_{channel_pair}.pdf", bbox_inches="tight", dpi=600)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare selected 2D channels from a folder, run pySODA, and combine results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--channels", nargs="+", type=int, required=True, help="Original input channels to keep, in output order.")
    parser.add_argument("--pattern", default="*", help="Input filename glob.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--time-index", type=int, default=0, help="Time frame to use for timelapse CZI/OME-TIFF inputs.")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Where selected/max-projected CYX TIFFs are written.")
    parser.add_argument("--overwrite-prepared", action="store_true")
    parser.add_argument("--roi-thresh", type=float, default=2.0)
    parser.add_argument("--channel-mask", type=int, default=None)
    parser.add_argument("--scale-list", nargs="+", default=["3,4"])
    parser.add_argument("--scale-threshold", nargs="+", default=["2.0"])
    parser.add_argument("--min-size", nargs="+", default=["10"])
    parser.add_argument("--min-axis", nargs="+", default=["3"])
    parser.add_argument("--min-intensity", nargs="+", default=["0"])
    parser.add_argument("--n-rings", type=int, default=20)
    parser.add_argument("--ring-width", type=int, default=1)
    parser.add_argument("--self-soda", action="store_true")
    parser.add_argument("--save-roi", action="store_true")
    parser.add_argument("--write-hist", dest="write_hist", action="store_true", default=True)
    parser.add_argument("--no-write-hist", dest="write_hist", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_time = time.time()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = args.prepared_dir or output_dir / "prepared_inputs"
    if args.overwrite_prepared and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    prepared_files = prepare_inputs(
        args.input_dir,
        prepared_dir,
        args.channels,
        args.pattern,
        args.recursive,
        args.time_index,
    )
    if not prepared_files:
        raise FileNotFoundError(f"No CZI/TIF/TIFF files found in {args.input_dir} with pattern {args.pattern}.")

    params = {
        "scale_list": _expand_scale_list(args.scale_list, len(args.channels)),
        "scale_threshold": _expand_values(args.scale_threshold, float, len(args.channels), "--scale-threshold"),
        "min_size": _expand_values(args.min_size, int, len(args.channels), "--min-size"),
        "min_axis": _expand_values(args.min_axis, int, len(args.channels), "--min-axis"),
        "min_intensity": _expand_values(args.min_intensity, float, len(args.channels), "--min-intensity"),
        "roi_thresh": args.roi_thresh,
        "channel_mask": args.channel_mask,
        "remove_channel": None,
        "n_rings": args.n_rings,
        "ring_width": args.ring_width,
        "self_soda": args.self_soda,
        "save_roi": args.save_roi,
        "write_hist": args.write_hist,
    }
    steps_SODA.main(str(prepared_dir), str(output_dir), params)
    write_combined_tables(output_dir, prepared_dir)
    print("--- Running time: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
