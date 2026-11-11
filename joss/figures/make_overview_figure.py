"""Overview figure for the JOSS paper: the database (top) and one target reduced end to end (bottom).

Run with an environment that has spherical 3.2.0 installed, for example
``~/data/sphere/envs/spherical-3.2.0/bin/python joss/figures/make_overview_figure.py``.
Writes ``joss/overview_figure.pdf`` and a PNG preview next to it.
"""

import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib.colors import AsinhNorm, FuncNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

SPHERE = Path.home() / "data/sphere"
DATABASE = SPHERE / "db_release_3.2.0_final"  # identical to Zenodo record 10.5281/zenodo.23884922
IRDIS_RAW = SPHERE / "data/IRDIS/science/*_51_Eri/DB_K12/2015-09-24/CORO"
IRDIS_CONVERTED = SPHERE / "reduction/IRDIS/observation/*_51_Eri/DB_K12/2015-09-24/converted"
IRDIS_TRAP = SPHERE / "reduction/IRDIS/trap/*_51_Eri/DB_K12/2015-09-24/template_matching"
IFS_TRAP = SPHERE / "spectral_library_v3.2_collected/IFS/trap/*_51_Eri/OBS_H/2015-09-24/template_matching"
OUTPUT = Path(__file__).resolve().parent.parent / ("overview_figure_galactic" if "--galactic" in sys.argv
                                                   else "overview_figure")

IRDIS_PIXEL_SCALE = 0.01225  # arcsec per pixel
TARGET = "51 Eri"

# Compact young regions, matched on MOCA association names and labelled at the median position
# of their sequences, with a label offset in points. Moving groups such as beta Pic are spread over
# the whole sky and are left unlabelled.
# Offsets are (equatorial, galactic).
REGIONS = {
    "Sco-Cen": (r"Sco-Cen|Scorpius-Centaurus|Centaurus Crux|Upper Scorpius|Upper Centaurus|Lupus",
                [(-6, -16), (-6, -2)]),
    "Taurus": (r"Taurus", [(6, 6), (6, 2)]),
    "Orion": (r"Orion", [(5, -12), (-4, 8)]),
    "Chamaeleon": (r"Chamaeleon", [(6, 2), (-60, -3)]),
    "TW Hya": (r"TW Hya", [(-4, 8), (-4, 8)]),
    "CrA": (r"Corona Australis", [(4, -10), (4, -10)]),
}
TARGET_LABEL_OFFSET = [(9, -2), (-36, -12)]
FIELD_HALF_WIDTH = 0.65  # arcsec, panels c and d

# Categorical slots 1-3 validated all-pairs for colour-vision deficiency; SAM is the
# fourth class and carries a different marker shape instead of a fourth hue.
MODES = {
    "IRDIS imaging": dict(tables=["irdis"], color="#eb6834", marker="o"),
    "IFS": dict(tables=["ifs"], color="#2a78d6", marker="o"),
    "IRDIS polarimetry": dict(tables=["irdis_polarimetry"], color="#1baf7a", marker="o"),
    "SAM": dict(tables=["ifs_sam", "irdis_sam"], color="#333333", marker="^"),
}
INK = "#222222"
# Links the target on the sky map to its name in the search code; no observing mode uses this hue.
TARGET_COLOR = "#b03fa8"  # passes the colour-vision checks against the three mode colours
MUTED = "#6b6b6b"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica Neue",
        "mathtext.tt": "Menlo",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "axes.titlepad": 5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def one(pattern):
    matches = sorted(glob(str(pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {pattern}, found {len(matches)}")
    return Path(matches[0])


def load_sequences():
    frames = []
    for mode, spec in MODES.items():
        for table in spec["tables"]:
            data = fits.getdata(DATABASE / f"table_of_observations_{table}.fits", 1)
            frames.append(
                pd.DataFrame(
                    {
                        "mode": mode,
                        "ra": np.asarray(data["RA_DEG"], float),
                        "dec": np.asarray(data["DEC_DEG"], float),
                        "year": pd.to_datetime(np.asarray(data["NIGHT_START"]).astype(str)).year,
                        "main_id": np.char.strip(np.asarray(data["MAIN_ID"]).astype(str)),
                        "association": np.asarray(data["MOCA_ASSOCIATION_NAME"]).astype(str),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


GALACTIC = "--galactic" in sys.argv


def sky_coordinates(ra_deg, dec_deg):
    """Mollweide axes coordinates with longitude increasing to the left."""
    lon, lat = np.asarray(ra_deg, float), np.asarray(dec_deg, float)
    if GALACTIC:
        galactic = SkyCoord(ra=lon * u.deg, dec=lat * u.deg).galactic
        lon, lat = galactic.l.deg, galactic.b.deg
    lon = np.radians(lon)
    lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
    return -lon, np.radians(lat)


def shade_galactic_plane(ax):
    longitude = np.linspace(-np.pi, np.pi, 400)
    for half_width, alpha in [(20, 0.035), (12, 0.05), (6, 0.07), (2.5, 0.09)]:
        b = np.radians(half_width)
        ax.fill_between(longitude, -b, b, color="#7a7a7a", alpha=alpha, linewidth=0, zorder=0)


def mode_legend(fig, sequences, y):
    handles = [
        Line2D([], [], marker="s" if spec["marker"] == "o" else "^", linestyle="", color=spec["color"], markersize=5,
               label=f"{mode} ({(sequences['mode'] == mode).sum()})")
        for mode, spec in MODES.items()
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.005, y), borderaxespad=0, ncol=4, frameon=False,
               handletextpad=0.3, columnspacing=1.2)


def draw_galactic_plane(ax):
    plane = SkyCoord(l=np.linspace(0, 360, 721) * u.deg, b=np.zeros(721) * u.deg, frame="galactic").icrs
    x, y = sky_coordinates(plane.ra.deg, plane.dec.deg)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ax.plot(x, y, color="#9a9a9a", linewidth=0.8, zorder=1)
    top = np.argmax(y)
    ax.annotate("Galactic plane", (x[top], y[top]), xytext=(0, 3), textcoords="offset points", fontsize=6.5,
                color=MUTED, ha="center", va="bottom",
                path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])


NORTHERN_LIMIT_DEC = 35  # deg; the northernmost sequence in the v3.2.0 tables is at +34.1


def shade_unobserved_north(ax):
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 361), np.linspace(-np.pi / 2, np.pi / 2, 181))
    declination = SkyCoord(l=-np.degrees(x) * u.deg, b=np.degrees(y) * u.deg, frame="galactic").icrs.dec.deg
    ax.contourf(x, y, declination, levels=[NORTHERN_LIMIT_DEC, 91], colors=["#bdbdbd"], alpha=0.35, zorder=0)
    ax.contour(x, y, declination, levels=[NORTHERN_LIMIT_DEC], colors=["#9a9a9a"], linewidths=0.6, zorder=1)
    pole = SkyCoord(ra=0 * u.deg, dec=90 * u.deg).galactic
    px, py = -np.radians(pole.l.deg - 360 if pole.l.deg > 180 else pole.l.deg), np.radians(pole.b.deg)
    ax.text(px, py, f"Dec > +{NORTHERN_LIMIT_DEC}°\nno SPHERE data", fontsize=6.5, color=MUTED, ha="center",
            va="center")


def plot_sky(ax, sequences):
    if GALACTIC:
        shade_galactic_plane(ax)
        shade_unobserved_north(ax)
    else:
        draw_galactic_plane(ax)
    for mode, spec in MODES.items():
        rows = sequences[sequences["mode"] == mode]
        x, y = sky_coordinates(rows["ra"], rows["dec"])
        ax.scatter(x, y, s=1.2 if spec["marker"] == "o" else 4, c=spec["color"], marker=spec["marker"],
                   linewidths=0, alpha=0.55, rasterized=True)
    target = sequences[sequences["main_id"].str.replace(r"\s+", " ", regex=True).str.endswith(TARGET)]
    x, y = sky_coordinates(target["ra"].iloc[:1], target["dec"].iloc[:1])
    ax.scatter(x, y, s=60, facecolors="none", edgecolors=TARGET_COLOR, linewidths=1.4, zorder=5)
    ax.annotate(TARGET, (x[0], y[0]), xytext=TARGET_LABEL_OFFSET[GALACTIC], textcoords="offset points", fontsize=6.5,
                color=TARGET_COLOR, fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
    for label, (pattern, offsets) in REGIONS.items():
        offset = offsets[GALACTIC]
        members = sequences[sequences["association"].str.contains(pattern, regex=True)]
        x, y = sky_coordinates([members["ra"].median()], [members["dec"].median()])
        ax.annotate(label, (x[0], y[0]), xytext=offset, textcoords="offset points", fontsize=6.5, color=INK,
                    fontstyle="italic", path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])
    ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
    ax.set_xticklabels([])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["−60°", "−30°", "0°", "+30°", "+60°"], fontsize=6.5, color=MUTED)

    ax.grid(True, color="#d9d9d9", linewidth=0.4)



def plot_years(ax, sequences):
    years = np.arange(sequences["year"].min(), sequences["year"].max() + 1)
    bottom = np.zeros(len(years))
    for mode, spec in MODES.items():
        counts = sequences[sequences["mode"] == mode].groupby("year").size().reindex(years, fill_value=0).to_numpy()
        ax.bar(years, counts, bottom=bottom, width=0.8, color=spec["color"], edgecolor="white", linewidth=0.5)
        bottom += counts
    ax.set_xticks(years[::2])
    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.set_ylabel("Sequences per year")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.4)
    ax.set_axisbelow(True)



def faint_structure_stretch(image):
    """Saturate the stellar core and stretch the halo, so speckles and satellite spots stand out."""
    low, high = np.nanpercentile(image, [2, 97])
    return AsinhNorm(linear_width=0.03 * (high - low), vmin=low, vmax=high)


def raw_k1_frame(half_width):
    path = sorted(glob(str(IRDIS_RAW / "*.fits")))[0]
    frame = fits.getdata(path)[0, :, :1024].astype(float)
    background = np.nanmedian(frame)
    filled = np.where(np.isfinite(frame), frame, background)
    core = filled[200:900, 200:900]
    kernel = np.ones(25) / 25
    blurred = np.apply_along_axis(np.convolve, 1, core, kernel, "same")
    blurred = np.apply_along_axis(np.convolve, 0, blurred, kernel, "same")
    y0, x0 = np.unravel_index(np.argmax(blurred), blurred.shape)
    y0, x0 = y0 + 200, x0 + 200
    return frame[y0 - half_width:y0 + half_width + 1, x0 - half_width:x0 + half_width + 1] - background


def plot_raw(ax):
    half_width = int(round(FIELD_HALF_WIDTH / IRDIS_PIXEL_SCALE))
    cutout = raw_k1_frame(half_width)
    extent = np.array([-half_width - 0.5, half_width + 0.5] * 2) * IRDIS_PIXEL_SCALE
    ax.imshow(cutout, origin="lower", extent=extent, cmap="gray", norm=faint_structure_stretch(cutout))
    ax.set_xticks([])
    ax.set_yticks([])
    bar_x = extent[0] + 0.08
    bar_y = extent[2] + 0.08
    ax.plot([bar_x, bar_x + 0.5], [bar_y, bar_y], color="white", linewidth=1.2)
    ax.text(bar_x + 0.25, bar_y + 0.03, "0.5″", color="white", fontsize=6.5, ha="center", va="bottom")



def plot_centring(ax):
    """First CENTER frame in K1, with the fitted satellite spots and the star centre derived from them."""
    converted = one(IRDIS_CONVERTED)
    frame = fits.getdata(converted / "center_cube.fits", memmap=True)[0, 0].astype(float)
    x0, y0 = fits.getdata(converted / "image_centers.fits")[0, 0]
    spots = fits.getdata(converted / "additional_outputs" / "spot_centers.fits")[0, 0]
    half_width = int(round(FIELD_HALF_WIDTH / IRDIS_PIXEL_SCALE))
    xc, yc = int(round(x0)), int(round(y0))
    cutout = frame[yc - half_width:yc + half_width + 1, xc - half_width:xc + half_width + 1]
    extent = np.array([xc - half_width - 0.5 - x0, xc + half_width + 0.5 - x0,
                       yc - half_width - 0.5 - y0, yc + half_width + 0.5 - y0]) * IRDIS_PIXEL_SCALE
    ax.imshow(cutout, origin="lower", extent=extent, cmap="gray", norm=faint_structure_stretch(cutout))
    for spot_x, spot_y in spots:
        ax.add_patch(Circle(((spot_x - x0) * IRDIS_PIXEL_SCALE, (spot_y - y0) * IRDIS_PIXEL_SCALE), 0.07, fill=False,
                            edgecolor="#eb6834", linewidth=0.9))
    ax.plot([0], [0], marker="+", markersize=9, markeredgewidth=1.2, color="#eb6834")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_detection(ax, cax, companion):
    # Channel 0 is K1. The planet is not detected in K2 (methane absorption), so combining channels only adds noise.
    data = fits.getdata(one(IRDIS_TRAP.parent / "norm_detection_ncomp*_temporal.fits"))[0]
    half = (data.shape[0] - 1) / 2
    # Pixel x grows to the west and ΔRA to the east, so east ends up on the left.
    extent = [(half + 0.5) * IRDIS_PIXEL_SCALE, -(half + 0.5) * IRDIS_PIXEL_SCALE,
              -(half + 0.5) * IRDIS_PIXEL_SCALE, (half + 0.5) * IRDIS_PIXEL_SCALE]
    # Symmetric power-law stretch: |S/N| < 3 stays pale, so noise recedes and only the planet saturates.
    power = 1.8
    stretch = FuncNorm((lambda v: np.sign(v) * np.abs(v) ** power, lambda v: np.sign(v) * np.abs(v) ** (1 / power)),
                       vmin=-5.5, vmax=5.5)
    image = ax.imshow(data, origin="lower", extent=extent, cmap="RdBu_r", norm=stretch)
    d_ra = -companion["x_relative"] * IRDIS_PIXEL_SCALE
    d_dec = companion["y_relative"] * IRDIS_PIXEL_SCALE
    ax.add_patch(Circle((d_ra, d_dec), 0.08, fill=False, edgecolor=TARGET_COLOR, linewidth=1.2))
    ax.annotate("51 Eri b", (d_ra, d_dec), xytext=(d_ra - 0.11, d_dec - 0.02), fontsize=6.5, color=TARGET_COLOR,
                fontweight="bold", va="center",
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlim(FIELD_HALF_WIDTH, -FIELD_HALF_WIDTH)
    ax.set_ylim(-FIELD_HALF_WIDTH, FIELD_HALF_WIDTH)
    ax.set_xlabel("ΔRA (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_title("Detection map, K1", loc="center", fontsize=6.5, color=MUTED, pad=3)
    colorbar = plt.colorbar(image, cax=cax)
    colorbar.set_ticks([-5, -3, 0, 3, 5])
    colorbar.set_label("Signal-to-noise ratio", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6.5)


def plot_spectrum(ax):
    ifs = pd.read_csv(one(IFS_TRAP / "overall_validated_companion_detections_spectra.csv")).dropna(subset=["contrast"])
    irdis = pd.read_csv(one(IRDIS_TRAP / "overall_validated_companion_detections_spectra.csv"))
    scale = 1e6
    ax.axhline(0, color="#bdbdbd", linewidth=0.5)
    ax.errorbar(ifs["wavelength"], ifs["contrast"] * scale, yerr=ifs["uncertainty"] * scale, fmt="o", markersize=2.2,
                color=MODES["IFS"]["color"], elinewidth=0.6, capsize=0, label="IFS YJH")
    ax.errorbar(irdis["wavelength"], irdis["contrast"] * scale, yerr=irdis["uncertainty"] * scale, fmt="s",
                markersize=3.2, color=MODES["IRDIS imaging"]["color"], elinewidth=0.8, capsize=0, label="IRDIS K1, K2")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Planet/star contrast (10$^{-6}$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False, handletextpad=0.3)
    ax.set_title("Spectrum of 51 Eri b", loc="center", fontsize=6.5, color=MUTED, pad=3)


def code_box(fig, x, y, text, fontsize=6.5, ha="left", highlight=None, highlight_color=None):
    """Dark code box. `highlight` is a substring drawn in `highlight_color` at its place in the code."""
    shown = text.replace(highlight, " " * len(highlight)) if highlight else text
    box = fig.text(x, y, shown, family="monospace", fontsize=fontsize, color="#f2f2f2", va="center", ha=ha,
                   multialignment="left", bbox=dict(boxstyle="round,pad=0.45", facecolor="#262626", edgecolor="none"))
    if not highlight:
        return
    # Every character of a monospace font has the same width, so the substring position follows from its column.
    lines = text.split("\n")
    row = next(index for index, line in enumerate(lines) if highlight in line)
    column = lines[row].index(highlight)
    extent = box.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    character_width = extent.width / max(len(line) for line in lines)
    line_height = extent.height / len(lines)
    fig.text(extent.x0 + column * character_width, extent.y1 - (row + 0.5) * line_height, highlight,
             family="monospace", fontsize=fontsize, color=highlight_color, fontweight="bold", va="center", ha="left")


def stage_heading(fig, y, stages):
    """Bold stage names joined by arrows, on one baseline."""
    renderer = fig.canvas.get_renderer()
    x = 0.015
    for index, stage in enumerate(stages):
        if index:
            # Helvetica has no arrow glyph.
            arrow = fig.text(x, y, "→", fontsize=7.5, va="baseline", color=MUTED, family="DejaVu Sans")
            x += arrow.get_window_extent(renderer).width / fig.bbox.width + 0.008
        text = fig.text(x, y, stage, fontsize=7.5, fontweight="bold", va="baseline", color=INK)
        x += text.get_window_extent(renderer).width / fig.bbox.width + 0.008


DATABASE_SHADE = "#eaf1f8"
SEARCH_SHADE = "#edf4ee"
WORKFLOW_SHADE = "#f4f2ee"


def section_box(fig, y0, y1, color):
    fig.patches.append(FancyBboxPatch((0.006, y0), 0.988, y1 - y0, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=-10))


PACKAGE = r"$\mathtt{spherical}$"  # package name in monospace, set through mathtext


def panel_title(fig, ax, y, text):
    position = ax.get_position()
    fig.text((position.x0 + position.x1) / 2, y, text, fontsize=7.5, va="baseline", ha="center", color=INK)


def main():
    sequences = load_sequences()
    companion = pd.read_csv(one(IRDIS_TRAP / "overall_validated_companion_detections.csv")).iloc[0]

    title_space = 0.28
    width, height = 6.3, 4.99 + title_space
    # Text is measured to place the highlighted target name, so measure at the resolution the figure is saved at.
    fig = plt.figure(figsize=(width, height), dpi=300)

    def y(inches_from_top):
        """Figure fraction for a distance from the top of the panels, below the figure title."""
        return 1 - (inches_from_top + title_space) / height

    def axes(x0, x_width, top, box_height, **kwargs):
        return fig.add_axes([x0, y(top + box_height), x_width, box_height / height], **kwargs)

    fig.text(0.5, 1 - 0.20 / height, f"The {PACKAGE} workflow, from the SPHERE archive to a planet spectrum",
             ha="center", va="baseline", fontsize=9, fontweight="bold", color=INK)

    section_box(fig, y(2.17), y(0.0), DATABASE_SHADE)
    fig.text(0.015, y(0.16), f"Database of all observations, curated by {PACKAGE} and published on Zenodo",
             va="baseline", fontsize=7.5, fontweight="bold", color=INK)
    mode_legend(fig, sequences, y(0.31))
    ax_sky = axes(0.02, 0.58, 0.56, 1.56, projection="mollweide")
    ax_years = axes(0.685, 0.29, 0.585, 1.27)
    panel_title(fig, ax_sky, y(0.50), "Sky position of every sequence" + (" (galactic, l = 0° at centre)" if GALACTIC else ""))
    panel_title(fig, ax_years, y(0.50), "Sequences per year")
    plot_sky(ax_sky, sequences)
    plot_years(ax_years, sequences)

    section_box(fig, y(2.74), y(2.21), SEARCH_SHADE)
    stage_heading(fig, y(2.35), ["Search and Filter Observations"])
    code_box(fig, 0.5, y(2.555),
             'db = SphereDatabase(table_of_observations, table_of_files, instrument="irdis")\n'
             'observations = db.retrieve_observation_metadata(db.filter(target_list=["51 Eri"], HCI_READY=True))',
             ha="center", highlight='"51 Eri"', highlight_color=TARGET_COLOR)

    section_box(fig, y(4.97), y(2.78), WORKFLOW_SHADE)
    stage_heading(fig, y(2.93), ["Automatic Data Reduction and Analysis"])
    code_box(fig, 0.5, y(3.12), "execute_targets(observations, config)", ha="center")

    panels_top = 3.53
    image_width = 0.165
    ax_raw = axes(0.025, image_width, panels_top, image_width * width)
    ax_centre = axes(0.205, image_width, panels_top, image_width * width)
    ax_det = axes(0.455, image_width, panels_top, image_width * width)
    cax = axes(0.455 + image_width + 0.008, 0.011, panels_top, image_width * width)
    ax_spec = axes(0.785, 0.2, panels_top, 1.0)
    plot_raw(ax_raw)
    plot_centring(ax_centre)
    ax_raw.set_title("Raw coronagraphic frame", loc="center", fontsize=6.5, color=MUTED, pad=3)
    ax_centre.set_title("Centering frame", loc="center", fontsize=6.5, color=MUTED, pad=3)
    plot_detection(ax_det, cax, companion)
    plot_spectrum(ax_spec)
    for ax in (ax_years, ax_spec):
        ax.set_facecolor("none")

    step_y = y(3.37)
    for text, left, right in [("1  Download", 0.025, 0.025 + image_width),
                              ("2  Pre-processing", 0.205, 0.205 + image_width),
                              ("3  Post-processing and detection", 0.405, 0.985)]:
        fig.text((left + right) / 2, step_y, text, fontsize=7.5, fontweight="bold", ha="center", va="bottom",
                 color=INK, linespacing=1.1)
    fig.add_artist(Line2D([0.415, 0.985], [step_y - 0.008, step_y - 0.008], transform=fig.transFigure,
                          color="#bdbdbd", linewidth=0.6))

    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)


if __name__ == "__main__":
    main()
