"""Compare the candidates of TRAP experiments and evaluate TRAP's shape check on them.

For each experiment folder (or bare TRAP result folder) the script reads the per-template
`companion_table_*.csv`, labels each candidate as 51 Eri b, one of the injected fakes or
"other", and decides for several sets of validation rules whether it passes:

- `trap`: TRAP's rule, reproduced (and checked against TRAP's validated table).
- `canonical`: the same limits applied to the ellipse in a form that does not depend on
  how the free Gaussian fit parametrised it (axes swapped, orientation modulo 180 deg).
- `canonical_35`, `canonical_45`: canonical with a wider angle limit.
- `elong_1.0`: canonical angle limit, any elongation >= 1.0.
- `no_shape`: SNR and good-pixel fraction only.

    python analyse_experiments.py $SPHERICAL_TUTORIAL_DIR/experiments/*/* --out summary
    python analyse_experiments.py ... --stamps stamps.png   # needs numpy, astropy, matplotlib

Writes `<out>_candidates.csv` and `<out>_summary.csv` and prints the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# 51 Eri b per instrument and night: separation in px and PA in deg.
# 2015: GRAVITY (tests/regression/data/51eri_astrometry_benchmark.md, section 3).
# 2017: Maire et al. 2019, Table A.1, K1 TLOCI, 449.0 mas at 12.25 mas/px.
REFERENCES = {
    ("IRDIS", "2015-09-24"): (455.364 / 12.25, 166.839),
    ("IRDIS", "2017-09-27"): (449.0 / 12.25, 155.67),
    ("IFS", "2015-09-24"): (455.364 / 7.46, 166.839),
}
MATCH_RADIUS = 3.0  # px


def fold180(angle: float) -> float:
    """Angle difference for an axis (period 180 deg), in [-90, 90)."""
    return (angle + 90.0) % 180.0 - 90.0


def trap_theta_deviation(position_angle: float, theta: float) -> float:
    """TRAP's `subtract_angles(position_angle, theta_free)`, period 360 deg."""
    return math.fmod((position_angle - theta) + 540.0, 360.0) - 180.0


def shape_metrics(position_angle, theta, x_fwhm, y_fwhm) -> dict:
    """TRAP's shape numbers and their parametrisation-free versions.

    TRAP expects the fitted Gaussian's y axis (the long one) at theta = position angle.
    An x-long fit at theta + 90 deg, or the same ellipse at theta + 180 deg, describes the
    same blob; `canonical_dev` and `elongation` are the same for all of them.
    """
    ratio = y_fwhm / x_fwhm
    if ratio >= 1.0:
        canonical_dev, elongation = fold180(position_angle - theta), ratio
    else:
        canonical_dev, elongation = fold180(position_angle - theta - 90.0), 1.0 / ratio
    return {"trap_dev": trap_theta_deviation(position_angle, theta), "ratio": ratio,
            "canonical_dev": canonical_dev, "elongation": elongation}


def rule_sets(detection_threshold=5.0, good_fraction=0.05, angle=25.0, ratio=(1.1, 4.5)):
    """Validation rules by name; each takes a candidate dict and returns a bool."""
    def base(c):
        return c["snr"] > detection_threshold and c["good_fraction"] > good_fraction

    def canonical(limit, low):
        return lambda c: base(c) and abs(c["canonical_dev"]) < limit and low < c["elongation"] < ratio[1]

    return {
        "trap": lambda c: base(c) and abs(c["trap_dev"]) < angle and ratio[0] < c["ratio"] < ratio[1],
        "canonical": canonical(angle, ratio[0]),
        "canonical_35": canonical(35.0, ratio[0]),
        "canonical_45": canonical(45.0, ratio[0]),
        "elong_1.0": canonical(angle, 1.0 - 1e-9),
        "no_shape": base,
    }


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_candidates(table: Path) -> list[dict]:
    """One dict per candidate of a `companion_table_<template>.csv` (rows repeat per channel)."""
    seen = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cid = row.get("candidate_id") or str(len(seen))
            if cid in seen:
                continue
            pa = _float(row["position_angle"])
            candidate = {
                "candidate_id": cid, "x_relative": _float(row["x_relative"]),
                "y_relative": _float(row["y_relative"]), "separation": _float(row["separation"]),
                "position_angle": pa,
                "snr": max(_float(row["norm_snr_fit_free"]), _float(row["peak_pixel_snr"])),
                "norm_snr_fit_free": _float(row["norm_snr_fit_free"]),
                "peak_pixel_snr": _float(row["peak_pixel_snr"]),
                "good_fraction": _float(row["good_fraction_free"]),
                "theta_free": _float(row["theta_free"]), "x_fwhm": _float(row["x_fwhm_free"]),
                "y_fwhm": _float(row["y_fwhm_free"]),
                "trap_theta_deviation": _float(row.get("theta_deviation")),
            }
            candidate.update(shape_metrics(pa, candidate["theta_free"], candidate["x_fwhm"], candidate["y_fwhm"]))
            seen[cid] = candidate
    return list(seen.values())


def validated_ids(table: Path) -> set[str]:
    """Candidate ids in TRAP's `validated_companion_table_<template>.csv`, if it exists."""
    if not table.exists():
        return set()
    with table.open(newline="") as handle:
        return {row["candidate_id"] for row in csv.DictReader(handle)}


def targets(instrument: str, night: str, experiment: dict | None) -> list[tuple[str, tuple[float, float]]]:
    """Known positions as (label, (y, x)): 51 Eri b, then the injected fakes."""
    found = []
    reference = REFERENCES.get((instrument, night))
    if reference is not None:
        sep, pa = reference
        found.append(("b", (sep * math.cos(math.radians(pa)), -sep * math.sin(math.radians(pa)))))
    for fake in (experiment or {}).get("fakes") or []:
        found.append((f"fake{fake['id']}", tuple(fake["yx"])))
    return found


def label_candidate(candidate: dict, known: list[tuple[str, tuple[float, float]]]) -> str:
    best, best_distance = "other", MATCH_RADIUS
    for label, (y, x) in known:
        distance = math.hypot(candidate["y_relative"] - y, candidate["x_relative"] - x)
        if distance <= best_distance:
            best, best_distance = label, distance
    return best


def find_result_folders(path: Path) -> list[Path]:
    """TRAP result folders (those holding template_matching/) at or below `path`."""
    if (path / "template_matching").is_dir():
        return [path]
    return sorted(p.parent for p in path.glob("reduction/*/trap/*/*/*/template_matching"))


def experiment_of(path: Path) -> dict | None:
    for folder in (path, *path.parents):
        if (folder / "experiment.json").exists():
            return json.loads((folder / "experiment.json").read_text())
    return None


def analyse(paths: list[Path], rules: dict) -> tuple[list[dict], list[dict]]:
    """Per-candidate rows and one summary row per (experiment, template, rule set)."""
    rows, summary = [], []
    for path in paths:
        for result in find_result_folders(path):
            experiment = experiment_of(result)
            parts = result.parts
            instrument, night = parts[-5], parts[-1]
            name = experiment["name"] if experiment else result.parts[-7]  # the run label
            known = targets(instrument, night, experiment)
            n_fakes = len((experiment or {}).get("fakes") or [])
            for table in sorted((result / "template_matching").glob("companion_table_*.csv")):
                template = table.stem.removeprefix("companion_table_")
                trap_ids = validated_ids(table.with_name(f"validated_companion_table_{template}.csv"))
                candidates = read_candidates(table)
                for c in candidates:
                    c["label"] = label_candidate(c, known)
                    for rule, test in rules.items():
                        c[f"pass_{rule}"] = bool(test(c))
                    c["trap_validated"] = c["candidate_id"] in trap_ids
                    rows.append({"experiment": name, "template": template, **c})
                mismatch = sorted(c["candidate_id"] for c in candidates if c["pass_trap"] != c["trap_validated"])
                for rule in rules:
                    passed = [c for c in candidates if c[f"pass_{rule}"]]
                    summary.append({
                        "experiment": name, "template": template, "rule": rule,
                        "b_found": any(c["label"] == "b" for c in candidates),
                        "b_passes": any(c["label"] == "b" for c in passed),
                        "fakes_found": sum(c["label"].startswith("fake") for c in candidates),
                        "fakes_passing": sum(c["label"].startswith("fake") for c in passed),
                        "fakes_injected": n_fakes,
                        "others_passing": sum(c["label"] == "other" for c in passed),
                        "trap_rule_mismatch": " ".join(mismatch),
                    })
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], summary: list[dict], template: str = "T-type") -> None:
    print(f"Candidates matched to b or a fake ({template}):")
    print(f"{'experiment':34} {'label':6} {'sep':>6} {'PA':>6} {'snr':>5} {'trapdev':>7} {'ratio':>5} "
          f"{'candev':>6} {'elong':>5} {'gf':>4} trap")
    for r in rows:
        if r["template"] == template and r["label"] != "other":
            print(f"{r['experiment'][:34]:34} {r['label']:6} {r['separation']:6.2f} {r['position_angle']:6.1f} "
                  f"{r['snr']:5.2f} {r['trap_dev']:7.1f} {r['ratio']:5.2f} {r['canonical_dev']:6.1f} "
                  f"{r['elongation']:5.2f} {r['good_fraction']:4.2f} {'yes' if r['pass_trap'] else 'no'}")
    print(f"\nPer rule set ({template}): b passes | fakes passing/found/injected | others passing")
    for s in summary:
        if s["template"] == template:
            flag = f"  TRAP MISMATCH {s['trap_rule_mismatch']}" if s["trap_rule_mismatch"] and s["rule"] == "trap" else ""
            print(f"{s['experiment'][:34]:34} {s['rule']:13} {'yes' if s['b_passes'] else 'no ':3} | "
                  f"{s['fakes_passing']}/{s['fakes_found']}/{s['fakes_injected']} | {s['others_passing']}{flag}")


def plot_stamps(paths: list[Path], out: Path, half: int = 12) -> None:
    """Stamp of the T-type normalised detection map around b and each fake, per experiment."""
    import matplotlib
    import numpy as np
    from astropy.io import fits

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for path in paths:
        for result in find_result_folders(path):
            image_path = result / "template_matching" / "normalized_detection_image_T-type.fits"
            if not image_path.exists():
                continue
            experiment = experiment_of(result)
            image = fits.getdata(image_path)[0]
            name = experiment["name"] if experiment else result.parts[-7]
            for label, (y, x) in targets(result.parts[-5], result.parts[-1], experiment):
                panels.append((f"{name}\n{label}", image, (y, x)))
    if not panels:
        return
    ncols = 6
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, image, (y, x)) in zip(axes.flat, panels):
        cy, cx = np.array(image.shape) // 2
        py, px = int(round(cy + y)), int(round(cx + x))
        stamp = image[max(py - half, 0):py + half + 1, max(px - half, 0):px + half + 1]
        ax.imshow(stamp, origin="lower", cmap="magma", vmin=-3, vmax=7)
        r = -np.array([y, x]) / math.hypot(y, x)  # towards the star
        ax.arrow(half, half, r[1] * 7, r[0] * 7, color="cyan", width=0.2)
        ax.set_title(title, fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="experiment or TRAP result folders")
    parser.add_argument("--out", type=Path, default=Path("experiments"), help="prefix of the two CSVs")
    parser.add_argument("--detection-threshold", type=float, default=5.0)
    parser.add_argument("--template", default="T-type", help="template shown in the printed summary")
    parser.add_argument("--stamps", type=Path, default=None, help="also write a PNG of stamps")
    parser.add_argument("--require-fakes", type=float, default=None, metavar="FRACTION",
                        help="exit 1 unless every experiment with fakes finds at least this fraction")
    args = parser.parse_args(argv)

    rows, summary = analyse(args.paths, rule_sets(detection_threshold=args.detection_threshold))
    write_csv(args.out.with_name(args.out.name + "_candidates.csv"), rows)
    write_csv(args.out.with_name(args.out.name + "_summary.csv"), summary)
    print_summary(rows, summary, args.template)
    if args.stamps:
        plot_stamps(args.paths, args.stamps)
    if args.require_fakes is not None:
        short = [s for s in summary if s["template"] == args.template and s["rule"] == "no_shape"
                 and s["fakes_injected"] and s["fakes_found"] < args.require_fakes * s["fakes_injected"]]
        for s in short:
            print(f"{s['experiment']}: found {s['fakes_found']} of {s['fakes_injected']} fakes")
        if short or not any(s["fakes_injected"] for s in summary):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
