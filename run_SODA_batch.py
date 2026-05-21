import argparse
import time
from pathlib import Path

import steps_SODA


def _triplet(values, cast, name):
    if len(values) == 1:
        return [cast(values[0])] * 3
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} expects either 1 value or 3 channel values.")
    return [cast(value) for value in values]


def _scale_list(values):
    if len(values) == 1:
        channels = values * 3
    elif len(values) == 3:
        channels = values
    else:
        raise argparse.ArgumentTypeError("--scale-list expects either 1 value or 3 channel values.")
    return [[int(scale) for scale in channel.split(",") if scale] for channel in channels]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run 2D pySODA on every TIF/TIFF in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing 2D multichannel TIF/TIFF files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where pySODA outputs will be written.")
    parser.add_argument("--roi-thresh", type=float, default=2.0)
    parser.add_argument("--channel-mask", type=int, default=None, help="Channel used for ROI mask generation.")
    parser.add_argument("--remove-channel", type=int, default=None, help="Channel to exclude from SODA analysis.")
    parser.add_argument(
        "--scale-list",
        nargs="+",
        default=["3,4"],
        help="Wavelet scales. Use one value for all channels, e.g. 3,4, or three values, e.g. 3,4 3,4 3,4.",
    )
    parser.add_argument("--scale-threshold", nargs="+", default=["2.0"], help="One value for all channels or three channel values.")
    parser.add_argument("--min-size", nargs="+", default=["30"], help="One value for all channels or three channel values.")
    parser.add_argument("--min-axis", nargs="+", default=["3"], help="One value for all channels or three channel values.")
    parser.add_argument("--min-intensity", nargs="+", default=["0"], help="One value for all channels or three channel values.")
    parser.add_argument("--n-rings", type=int, default=20)
    parser.add_argument("--ring-width", type=int, default=1)
    parser.add_argument("--self-soda", action="store_true")
    parser.add_argument("--save-roi", action="store_true", help="Write QC TIF mask/spot images.")
    parser.add_argument("--write-hist", action="store_true", help="Write per-image PDF histograms.")
    return parser


def main():
    args = build_parser().parse_args()
    params = {
        "scale_list": _scale_list(args.scale_list),
        "scale_threshold": _triplet(args.scale_threshold, float, "--scale-threshold"),
        "min_size": _triplet(args.min_size, int, "--min-size"),
        "min_axis": _triplet(args.min_axis, int, "--min-axis"),
        "min_intensity": _triplet(args.min_intensity, float, "--min-intensity"),
        "roi_thresh": args.roi_thresh,
        "channel_mask": args.channel_mask,
        "remove_channel": args.remove_channel,
        "n_rings": args.n_rings,
        "ring_width": args.ring_width,
        "self_soda": args.self_soda,
        "save_roi": args.save_roi,
        "write_hist": args.write_hist,
    }

    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_SODA.main(str(args.input_dir), str(args.output_dir), params)
    print("--- Running time: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
