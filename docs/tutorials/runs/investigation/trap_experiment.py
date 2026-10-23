"""Run TRAP once with chosen settings on the products of a tutorial run, optionally with fake planets.

Used to find out why 51 Eri b fails TRAP's shape check in the 2015 IRDIS tutorial run
(see README.md next to this file). Each experiment gets its own folder,
`$SPHERICAL_TUTORIAL_DIR/experiments/<source>/<name>/`, whose observation tree is
hard-linked to the source run, so TRAP's results never replace the tutorial's. The
source files are read-only while TRAP runs and are compared before and after.

    python trap_experiment.py --source 51eri_irdis --name a31-43_f0.20_known \\
        --inner 31 --outer 43 --frac 0.20 --known=-35.95,-8.43 --ncpu 48

With `--inject SEP:PA:SCALE` (repeatable) the science cube of the experiment is replaced
by a copy with fake planets: the flux PSF times `--fake-spectrum` (one contrast per
channel) times SCALE, at separation SEP px and position angle PA deg, placed with TRAP's
own `makesource.addsource` and the centres, angles, anamorphism and amplitude modulation
that `run_trap` passes to TRAP. The copy is deleted after the run unless `--keep-cube`.

With `--resume` an experiment whose TRAP run did not finish its detection (for example
after a crash in template matching) is run again in place with the same settings. TRAP
keeps a finished reduction, so only the detection runs; fakes are injected again first,
since the detection reads the data.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
sys.path[:0] = [str(HERE), str(RUNS), str(RUNS.parent), str(RUNS.parents[1] / "tools")]
import annulus_timing  # noqa: E402
from tutorial_helpers import tutorial_dir  # noqa: E402


def relative_yx(separation: float, position_angle: float) -> tuple[float, float]:
    """Offset (y, x) in px from the star for a position angle east of north.

    North is +y and east is -x, the orientation TRAP's tables use
    (position_angle = atan2(-x, y)).
    """
    pa = math.radians(position_angle)
    return separation * math.cos(pa), -separation * math.sin(pa)


def parse_fake(spec: str) -> tuple[float, float, float]:
    """`SEP:PA:SCALE` to floats."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--inject expects SEP:PA:SCALE, got {spec!r}")
    return tuple(float(p) for p in parts)


def parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def inject_fakes(cube, psf, pa, yx_centers, fakes, yx_anamorphism, right_handed, amplitude=None):
    """Add fake planets to a copy of `cube`.

    cube (n_wave, n_frames, ny, nx); psf (n_wave, sy, sx); pa (n_frames,) deg;
    yx_centers (n_wave, n_frames, 2); fakes is a list of ((y, x), contrasts per channel);
    amplitude (n_wave, n_frames) or None. TRAP multiplies its planet model by the
    amplitude modulation, so the fakes get it too.
    """
    import numpy as np
    from trap.makesource import addsource

    out = np.array(cube, copy=True)
    for w in range(out.shape[0]):
        stamp = np.nan_to_num(np.asarray(psf[w], dtype=float))
        stamp[stamp < 0] = 0.0
        for yx, contrasts in fakes:
            norm = np.full(out.shape[1], float(contrasts[w]))
            if amplitude is not None:
                norm = norm * np.asarray(amplitude[w], dtype=float)
            out[w] = addsource(out[w], yx, np.asarray(pa, dtype=float), stamp,
                               image_center=np.asarray(yx_centers[w], dtype=float), norm=norm,
                               yx_anamorphism=yx_anamorphism, right_handed=right_handed,
                               subpixel=True, copy=False)
    return out


def replace_with_copy(path: Path, data, header) -> None:
    """Write `data` to `path` as a new file, leaving any hard-linked twin untouched."""
    from astropy.io import fits

    path.unlink()
    fits.writeto(path, data, header=header)


def inject_into_experiment(data_directory: Path, observation, reduction_config, trap_config,
                           fakes) -> dict:
    """Replace the science cube in `data_directory` with a copy carrying the fakes."""
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    from spherical.pipeline.run_trap import _instrument_of
    from spherical.pipeline.science_frames import normalize_centers_to_frames, science_frame_type

    continuous = bool(observation.observation["WAFFLE_MODE"][0])
    identifier = science_frame_type(continuous)
    pa = pd.read_csv(data_directory / f"frames_info_{identifier}.csv")["DEROT ANGLE"].values
    cube_path = data_directory / f"{identifier}_cube.fits"
    with fits.open(cube_path) as hdul:
        cube, header = hdul[0].data.copy(), hdul[0].header.copy()
    psf = np.nanmean(fits.getdata(data_directory / "psf_cube_for_postprocessing.fits"), axis=1)
    xy = fits.getdata(data_directory / "image_centers_fitted_robust.fits")
    xy = normalize_centers_to_frames(xy, len(pa), _instrument_of(observation), continuous)
    amplitude = None
    amp_path = data_directory / "spot_amplitude_variation.fits"
    if continuous and reduction_config.pass_amplitude_modulation_to_trap and amp_path.exists():
        amplitude = fits.getdata(amp_path)
    reduction = trap_config.reduction
    injected = inject_fakes(cube, psf, pa, np.asarray(xy)[..., ::-1], [(f["yx"], f["contrast"]) for f in fakes],
                            yx_anamorphism=np.asarray(reduction.yx_anamorphism, dtype=float),
                            right_handed=reduction.right_handed, amplitude=amplitude)
    replace_with_copy(cube_path, injected.astype(cube.dtype), header)
    return {"cube": cube_path.name, "amplitude_modulation": amplitude is not None, "frames": len(pa)}


SETTINGS = ("source", "inner", "outer", "frac", "known", "fakes")


def trap_folders(exp_dir: Path) -> list[Path]:
    """TRAP result folders of an experiment (one per observation)."""
    return sorted(p for p in exp_dir.glob("reduction/*/trap/*/*/*") if p.is_dir())


def detection_finished(exp_dir: Path) -> bool:
    folders = trap_folders(exp_dir)
    return bool(folders) and all((f / ".run_trap_detection.done").exists() for f in folders)


def restore_science_cubes(data_directory: Path, source_directory: Path) -> None:
    """Hard-link science cubes that an earlier injected run deleted back from the source."""
    for name in ("coro_cube.fits", "center_cube.fits"):
        if not (data_directory / name).exists() and (source_directory / name).exists():
            os.link(source_directory / name, data_directory / name)


def set_aside_crash_reports(exp_dir: Path) -> list[str]:
    """Rename earlier crash reports so a new one shows whether the rerun crashed too."""
    moved = []
    for report in exp_dir.glob("reduction/*/trap/*/*/*/trap_crash_report.txt"):
        target = report.with_name(f"trap_crash_report_before_resume_{len(moved)}.txt")
        report.rename(target)
        moved.append(target.relative_to(exp_dir).as_posix())
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="tutorial run label, e.g. 51eri_irdis")
    parser.add_argument("--name", required=True, help="experiment folder name")
    parser.add_argument("--ncpu", type=int, required=True)
    parser.add_argument("--inner", type=int, required=True, help="search_region_inner_bound, px")
    parser.add_argument("--outer", type=int, required=True, help="search_region_outer_bound, px")
    parser.add_argument("--frac", type=float, required=True, help="temporal_components_fraction")
    parser.add_argument("--known", type=parse_floats, default=None,
                        help="yx_known_companion_position as Y,X in px (use --known=-35.95,-8.43)")
    parser.add_argument("--inject", type=parse_fake, action="append", default=[],
                        help="fake planet SEP:PA:SCALE (px, deg, times --fake-spectrum)")
    parser.add_argument("--fake-spectrum", type=parse_floats, default=[6.95e-6, 1.09e-6],
                        help="contrast per channel at SCALE 1; default 51 Eri b in K1, K2 (regression)")
    parser.add_argument("--keep-cube", action="store_true", help="keep the injected science cube")
    parser.add_argument("--species-dir", type=Path, default=Path.home() / "data/sphere/species")
    parser.add_argument("--resume", action="store_true",
                        help="rerun an experiment whose detection did not finish, with the same settings")
    args = parser.parse_args(argv)

    import stamp_notebook

    from spherical.pipeline.run_trap import _data_directory_for, _instrument_of, run_trap_on_observations

    run = importlib.import_module(args.source)
    exp_dir = tutorial_dir() / "experiments" / args.source / args.name
    previous = None
    if exp_dir.exists():
        if not args.resume:
            print(f"{exp_dir} exists from an earlier run; remove it, choose another --name or pass --resume.")
            return 1
        if detection_finished(exp_dir):
            print(f"{exp_dir} has a finished detection; nothing to resume.")
            return 0
        if (exp_dir / "experiment.json").exists():
            previous = json.loads((exp_dir / "experiment.json").read_text())

    config, trap_config = run.build(args.ncpu, args.source)
    config.directories.reduction_directory = exp_dir / "reduction"
    reduction_overrides = {"search_region_inner_bound": args.inner, "search_region_outer_bound": args.outer}
    if args.known is not None:
        reduction_overrides["yx_known_companion_position"] = list(args.known)
    trap_config.reduction = trap_config.reduction.merge(**reduction_overrides)
    trap_config.processing = trap_config.processing.merge(temporal_components_fraction=[args.frac])

    database_dir = tutorial_dir() / "database" if (tutorial_dir() / "database").is_dir() else None
    _table, observations = run.template.select_observations(
        [run.runner.TARGET], database_directory=database_dir, OBS_ID=run.OBS_ID)
    if len(observations) != 1:
        print(f"Expected one observation for OBS_ID {run.OBS_ID}, found {len(observations)}.")
        return 1
    inst = _instrument_of(observations[0])
    src_obs = tutorial_dir() / args.source / "reduction" / inst / "observation"
    if not src_obs.is_dir():
        print(f"{src_obs} does not exist; run docs/tutorials/runs/{args.source}.py first.")
        return 1
    exp_obs = exp_dir / "reduction" / inst / "observation"

    fakes = []
    for k, (sep, pa, scale) in enumerate(args.inject):
        fakes.append({"id": k, "separation": sep, "position_angle": pa, "scale": scale,
                      "yx": list(relative_yx(sep, pa)), "contrast": [c * scale for c in args.fake_spectrum]})
    known = list(args.known) if args.known is not None else None
    settings = {"source": args.source, "inner": args.inner, "outer": args.outer, "frac": args.frac,
                "known": known, "fakes": fakes}
    if previous is not None:
        differ = [key for key in SETTINGS if json.loads(json.dumps(previous.get(key))) != json.loads(json.dumps(settings[key]))]
        if differ:
            print(f"{exp_dir} was run with other settings ({', '.join(differ)}); remove it to start again.")
            return 1

    data_directory = Path(_data_directory_for(inst, config, observations[0]))
    crash_reports = []
    if exp_obs.exists():
        restore_science_cubes(data_directory, src_obs / data_directory.relative_to(exp_obs))
        crash_reports = set_aside_crash_reports(exp_dir)
    else:
        annulus_timing.hardlink_tree(src_obs, exp_obs)
    injection = None
    if fakes:
        injection = inject_into_experiment(data_directory, observations[0], config, trap_config, fakes)

    load_before = os.getloadavg()
    before = annulus_timing.snapshot(src_obs)
    with annulus_timing.read_only(src_obs):
        start = time.monotonic()
        run_trap_on_observations(observations=observations, trap_config=trap_config,
                                 reduction_config=config, species_database_directory=args.species_dir)
        seconds = time.monotonic() - start
    changed = annulus_timing.changed_files(before, annulus_timing.snapshot(src_obs))
    if fakes and not args.keep_cube:
        (data_directory / injection["cube"]).unlink()

    crashed = any((f / "trap_crash_report.txt").exists() for f in trap_folders(exp_dir))
    run_record = {"seconds": round(seconds, 1), "ncpu": args.ncpu,
                  "load_before": [round(v, 1) for v in load_before],
                  "load_after": [round(v, 1) for v in os.getloadavg()],
                  "machine": stamp_notebook.collect_provenance(args.ncpu, datetime.date.today())}
    if previous is None:
        result = {"name": args.name, "obs_id": run.OBS_ID, "instrument": inst, **settings,
                  "injection": injection, **run_record}
    else:
        # The first run's time is the reduction's; a resume only adds the detection.
        result = previous
        result.setdefault("resumes", []).append({**run_record, "crash_reports_set_aside": crash_reports})
    result.update({"source_changed": changed, "crashed": crashed,
                   "detection_finished": detection_finished(exp_dir)})
    (exp_dir / "experiment.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.name}: TRAP took {seconds / 60:.1f} min with set_ncpu({args.ncpu}).")
    if changed:
        print(f"TRAP changed files of the source run: {changed}")
        return 1
    if crashed or not result["detection_finished"]:
        print(f"TRAP did not finish the detection; see trap_crash_report.txt under {exp_dir}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
