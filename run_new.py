"""
CLI runner for analysis_pipeline.
Keeps original scripts; uses argparse for flexible inputs.
"""
import argparse
import sys
from pathlib import Path

from analysis_pipeline import (
    adjust_ch01_image,
    run_full_analysis,
    generate_final_image,
)


def _check_exists(path_str, label):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image pipeline runner (mask -> axes -> quadrants).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_adj = sub.add_parser("adjust", help="Adjust ch01 image (exposure/brightness/contrast).")
    p_adj.add_argument("--input", required=True, help="Path to ch01 image.")
    p_adj.add_argument("--output", required=True, help="Output path for adjusted image.")
    p_adj.add_argument("--brightness", type=float, default=10.0, help="Brightness offset.")
    p_adj.add_argument("--exposure", type=float, default=0.3, help="Exposure multiplier.")
    p_adj.add_argument("--contrast", type=float, default=0.2, help="Contrast gain.")

    p_full = sub.add_parser("analyze", help="Full pipeline: mask -> axes -> quadrants.")
    p_full.add_argument("--ch00", required=True, help="Path to channel-0 image.")
    p_full.add_argument("--brightness", required=True, help="Path to brightness image (e.g., B.png).")
    p_full.add_argument("--mask-out", default="mask.png", help="Output mask/boundary image.")
    p_full.add_argument("--axes-out", default="axes_output.png", help="Output axes image.")
    p_full.add_argument("--csv-out", default="quadrant_results.csv", help="Output CSV file.")
    p_full.add_argument("--id-map-out", default="particle_id_map.png", help="Output ID map PNG.")

    p_comp = sub.add_parser("composite", help="Make A transparent and overlay on B.")
    p_comp.add_argument("--A", required=True, help="Path to A image.")
    p_comp.add_argument("--B", required=True, help="Path to B image.")
    p_comp.add_argument("--A1-out", default="A1.png", help="Output for transparent A.")
    p_comp.add_argument("--C1-out", default="C1.png", help="Output for composite.")

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        if args.cmd == "adjust":
            _check_exists(args.input, "input image")
            adjust_ch01_image(
                ch01_path=args.input,
                save_path=args.output,
                brightness=args.brightness,
                exposure=args.exposure,
                contrast_gain=args.contrast,
            )
            print(f"[OK] adjusted image saved to {args.output}")

        elif args.cmd == "analyze":
            _check_exists(args.ch00, "ch00 image")
            _check_exists(args.brightness, "brightness image")
            run_full_analysis(
                ch00_path=args.ch00,
                brightness_image_path=args.brightness,
                mask_out=args.mask_out,
                axes_out=args.axes_out,
                csv_out=args.csv_out,
                id_map_out=args.id_map_out,
            )
            print(f"[OK] analysis done. mask={args.mask_out}, axes={args.axes_out}, "
                  f"id_map={args.id_map_out}, csv={args.csv_out}")

        elif args.cmd == "composite":
            _check_exists(args.A, "A image")
            _check_exists(args.B, "B image")
            generate_final_image(
                A_path=args.A,
                B_path=args.B,
                A1_path=args.A1_out,
                C1_path=args.C1_out,
            )
            print(f"[OK] composite done. A1={args.A1_out}, C1={args.C1_out}")

        else:
            print("Unknown command", file=sys.stderr)
            return 1

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
