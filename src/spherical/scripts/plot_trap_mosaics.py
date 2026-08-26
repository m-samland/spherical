"""Command-line entry point for creating mosaics of TRAP results."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt

from spherical.database.paths import ENV_DATABASE_DIR, resolve_database_dir
from spherical.pipeline.visualize import mosaic

logger = logging.getLogger(__name__)

TEMPLATE_TYPES = ["flat", "L-type", "T-type"]
CONTENT_CHOICES = ["combined", "detection", "spectrum"]
INSTRUMENT_CHOICES = ["ifs", "ifs_sam", "irdis", "irdis_polarimetry", "irdis_sam"]
DEFAULT_INSTRUMENT = "ifs"

# content -> (single function name, batched function name, supports auto_scale)
_PLOT_DISPATCH = {
    "combined": ("plot_combined_mosaic", "plot_combined_mosaic_batched", True),
    "detection": ("plot_detection_mosaic", "plot_detection_mosaic_batched", True),
    "spectrum": ("plot_spectrum_mosaic", "plot_spectrum_mosaic_batched", False),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create mosaic summaries (PDF by default) of TRAP detection results."
    )
    parser.add_argument(
        "base_path",
        type=Path,
        help=(
            "TRAP results root (contains target/mode/date/...). "
            "Template-matching results are used when available; "
            "otherwise the regular TRAP detection map is used."
        ),
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=None,
        help="Directory holding table_of_observations_<instrument>.fits. Optional; "
        f"falls back to the ${ENV_DATABASE_DIR} environment variable when not given. "
        "Without either, titles omit exposure-time/rotation metadata.",
    )
    parser.add_argument(
        "--instrument",
        choices=INSTRUMENT_CHOICES,
        default=None,
        help="Which observation table to read metadata from. Default: inferred "
        f"from an IFS/IRDIS path segment in BASE_PATH, else '{DEFAULT_INSTRUMENT}'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: BASE_PATH/mosaics/).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Filename suffix so subset runs do not overwrite "
        "(e.g. combined_mosaic_flat_young.pdf).",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        choices=TEMPLATE_TYPES,
        default=list(TEMPLATE_TYPES),
        help="Template subset to plot (default: all three).",
    )
    parser.add_argument(
        "--content",
        choices=CONTENT_CHOICES,
        default="combined",
        help="What to plot (default: combined = detection maps + spectra).",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Output format (default: pdf).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Split into files of N observations each (default: single file). "
        "Batched filenames add a _batch_NN_of_MM marker and honor --format/--suffix.",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=None,
        help="Drop candidates with SNR below this value.",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=None,
        help="Drop candidates with SNR above this value.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300).",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Enable global auto color-scaling (default: off, fixed sigma limits). "
        "Ignored for --content spectrum.",
    )
    return parser


def resolve_output_dir(base_path: Path, output: Optional[Path]) -> Path:
    return output if output is not None else base_path / "mosaics"


def build_output_path(
    output_dir: Path, content: str, template: str, suffix: Optional[str], fmt: str
) -> Path:
    stem = f"{content}_mosaic_{template}"
    if suffix:
        stem += f"_{suffix}"
    return output_dir / f"{stem}.{fmt}"


def select_plot_function(content: str, batched: bool) -> Callable:
    single_name, batched_name, _ = _PLOT_DISPATCH[content]
    return getattr(mosaic, batched_name if batched else single_name)


def resolve_instrument(cli_value: Optional[str], base_path: Path) -> str:
    """Resolve which observation table to use: explicit ``--instrument`` wins,
    otherwise infer from an ``IFS``/``IRDIS`` segment in the results path.

    Reduction trees are laid out as ``{reduction_dir}/{IFS,IRDIS}/trap/...``, so
    the instrument is normally recoverable without the user restating it. The
    SAM and polarimetry variants share that segment and must be named explicitly.
    """
    if cli_value is not None:
        return cli_value
    segments = {part.upper() for part in base_path.resolve().parts}
    if "IRDIS" in segments and "IFS" not in segments:
        return "irdis"
    if "IFS" in segments and "IRDIS" not in segments:
        return "ifs"
    return DEFAULT_INSTRUMENT


def obs_table_filename(instrument: str) -> str:
    return f"table_of_observations_{instrument}.fits"


def load_obs_table(database_dir: Optional[Path], instrument: str = DEFAULT_INSTRUMENT):
    if database_dir is None:
        logger.warning(
            "No --database-dir given and $%s not set; titles will omit "
            "exposure-time/rotation metadata.",
            ENV_DATABASE_DIR,
        )
        return None
    table_path = database_dir / obs_table_filename(instrument)
    if not table_path.exists():
        logger.warning(
            "Observation table not found at %s; continuing without metadata.", table_path
        )
        return None
    return mosaic.load_observation_table(table_path)


def run(args: argparse.Namespace) -> int:
    if not args.base_path.is_dir():
        logger.error("Base path does not exist or is not a directory: %s", args.base_path)
        return 1

    combinations = mosaic.get_all_combinations(args.base_path)
    if not combinations:
        logger.warning("No target/mode/date combinations found under %s", args.base_path)
        return 1

    output_dir = resolve_output_dir(args.base_path, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    instrument = resolve_instrument(args.instrument, args.base_path)
    logger.info("Reading observation metadata for instrument '%s'", instrument)
    obs_table = load_obs_table(resolve_database_dir(args.database_dir), instrument)

    _, _, supports_auto_scale = _PLOT_DISPATCH[args.content]
    batched = args.batch_size is not None
    plot_fn = select_plot_function(args.content, batched)

    common = dict(
        observation_table=obs_table,
        dpi=args.dpi,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
    )
    if supports_auto_scale:
        common["auto_scale"] = args.auto_scale

    for template in args.templates:
        if batched:
            figures = plot_fn(
                args.base_path,
                template_type=template,
                batch_size=args.batch_size,
                output_dir=output_dir,
                output_format=args.format,
                suffix=args.suffix,
                **common,
            )
            for fig in figures:
                plt.close(fig)
            logger.info("Wrote batched %s mosaics for template '%s'", args.content, template)
        else:
            output_path = build_output_path(
                output_dir, args.content, template, args.suffix, args.format
            )
            fig = plot_fn(
                args.base_path, template_type=template, output_path=output_path, **common
            )
            plt.close(fig)
            logger.info("Wrote %s", output_path)

    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plt.switch_backend("Agg")
    args = build_parser().parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
