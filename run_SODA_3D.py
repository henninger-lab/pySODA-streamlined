from __future__ import annotations

import argparse
from pathlib import Path

from ome_3d import inspect_ome_tiff, load_ome_tiff_czyx, parse_crop
from steps_SODA_3D import run_3d_soda


DEFAULT_IMAGE = Path(
    "/Volumes/T7 Shield/260514_rbp_sync/radplots/"
    "260512_ST_CTD-Halo(far-red)_MED1-mStayGold_CBP-red-IF-06_processed_global3d_aligned.ome.tiff"
)
DEFAULT_MASK = Path(
    "/Volumes/T7 Shield/260514_rbp_sync/radplots/"
    "260518_rbp_syncon_cbp_if_radial_plots/2_segmentation/"
    "260512_ST_CTD-Halo(far-red)_MED1-mStayGold_CBP-red-IF-06_processed_global3d_aligned.ome_max_seg.tif"
)
DEFAULT_OUTPUT = Path(
    "/Volumes/T7 Shield/260514_rbp_sync/radplots/"
    "260518_rbp_syncon_cbp_if_radial_plots/pySODA_3D"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a true-3D SODA-style analysis on OME-TIFF SIM stacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focus-channel", type=int, default=0)
    parser.add_argument("--subject-channel", type=int, default=2)
    parser.add_argument("--focus-name", default="CTD")
    parser.add_argument("--subject-name", default="MED1")
    parser.add_argument("--log-sigma-zyx", type=float, nargs=3, default=(1.0, 2.0, 2.0))
    parser.add_argument("--log-method", choices=["dense", "separable"], default="separable")
    parser.add_argument("--truncate", type=float, default=4.0)
    parser.add_argument("--tile-yx", type=int, default=224)
    parser.add_argument("--threshold-multiplier", type=float, default=8.0)
    parser.add_argument("--min-volume", type=int, default=30)
    parser.add_argument("--shell-width-um", type=float, default=0.05)
    parser.add_argument("--n-shells", type=int, default=20)
    parser.add_argument("--random-iterations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--save-label-images",
        action="store_true",
        help="Also save full 3D spot-label TIFFs. These can be very large for full SIM stacks.",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Fail immediately unless PyTorch reports an available Apple MPS device.",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional cropped validation region formatted as z0:z1,y0:y1,x0:x1.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print OME-TIFF metadata and validate optional crop loading without running spot detection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    crop = parse_crop(args.crop)

    if args.inspect_only:
        info = inspect_ome_tiff(args.image)
        print(info)
        data, axes, _ = load_ome_tiff_czyx(args.image, crop=crop)
        print(f"Loaded shape={data.shape}, dtype={data.dtype}, axes={axes}")
        return

    summary = run_3d_soda(
        image_path=args.image,
        mask_path=args.mask,
        output_dir=args.output_dir,
        focus_channel=args.focus_channel,
        subject_channel=args.subject_channel,
        focus_name=args.focus_name,
        subject_name=args.subject_name,
        log_sigma_zyx=tuple(args.log_sigma_zyx),
        log_method=args.log_method,
        truncate=args.truncate,
        tile_yx=args.tile_yx,
        threshold_multiplier=args.threshold_multiplier,
        min_volume=args.min_volume,
        shell_width_um=args.shell_width_um,
        n_shells=args.n_shells,
        random_iterations=args.random_iterations,
        chunk_size=args.chunk_size,
        crop=crop,
        random_seed=args.random_seed,
        save_label_images=args.save_label_images,
        require_mps=args.require_mps,
    )
    print("3D SODA complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
