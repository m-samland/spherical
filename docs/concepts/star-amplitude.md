---
stage: reduction
---

# Star amplitude

After reading this page you know how spherical measures how bright the star is in
every science frame, how it ties that measurement to the {term}`FLUX frame`s, and what
{term}`TRAP` does with it.
Two steps do the work.
`observing_conditions` collects the adaptive optics (AO) telemetry and the weather
data of the observation, and `calibrate_star_amplitude` turns the
{term}`WFS` flux in that telemetry into the {term}`Star amplitude`.
Both are on by default.

## Why TRAP needs the brightness of the star

TRAP models a companion with a {term}`Forward model` built from the stellar PSF,
which spherical measures in the FLUX frames before and after the coronagraphic
sequence.
The model assumes that the star, and with it any companion, is as bright in every
science frame as in the FLUX frames.

Thin clouds, changing seeing and changing AO performance alter how much starlight
reaches the detector during an hour-long sequence.
A companion in a frame that received 20% less light looks 20% fainter, so a fit that
ignores the change underestimates its {term}`Contrast`.
The science detector cannot measure this itself, because the coronagraph hides the
star.

spherical therefore gives TRAP a star amplitude for every science frame and
wavelength.
It is the brightness of the star relative to the PSF that TRAP uses, so 1 means as
bright as that PSF and 0.9 means 10% fainter.
TRAP scales the forward model of each frame by it.

## What the wavefront sensor measures

```{instrument-background}
SAXO, the extreme AO system of SPHERE, measures the wavefront with a visible
Shack-Hartmann {term}`WFS` at up to 1.38 kHz. A dichroic sends the visible light to
the WFS and the near-infrared light to IRDIS and IFS
([Beuzit et al. 2019](https://doi.org/10.1051/0004-6361/201935251);
[Sauvage et al. 2016](https://doi.org/10.1117/1.JATIS.2.2.025003)).
The real-time computer of SAXO, {term}`SPARTA`
([Fedrigo et al. 2006](https://doi.org/10.1117/12.671919)), records the flux of the WFS together with the Strehl ratio, seeing and coherence time it
estimates. ESO archives these records as files of their own, a few minutes each.
The infrared {term}`ND filter` of the FLUX frames sits in the near-infrared arm, so it
does not change the WFS flux.
```

The WFS flux therefore follows the starlight that enters SPHERE during every frame,
including the coronagraphic ones.
spherical uses it as the time series of the star amplitude.
This assumes that the flux in the science band changes in proportion to the visible
WFS flux, which holds for grey extinction by clouds.

## Observing conditions per frame

The `observing_conditions` step lists the SPARTA files of the observation in the ESO
archive (`https://archive.eso.org/tap_obs/sync`) and downloads them.
It reads the visible loop (WFS flux), the infrared loop, the differential tip-tilt
sensor (DTTS) and the atmospheric estimates of each file.
It also queries the Paranal ambient monitors MASS, DIMM, meteo and LHATPRO for the
time of the frames and ten minutes on either side (`margin_s`).

For every frame the step averages the samples that overlap its exposure, and records
the flag `direct`.
A frame without an overlapping sample gets a value interpolated from its neighbours,
flag `interpolated`, when both lie within `max_interp_gap_s` (60 s), and NaN with the
flag `gap` otherwise.
Samples from two SPARTA segments are never mixed in one frame.
SPARTA starts a new segment whenever the AO configuration changes.

The step needs a network connection.
It gives all its network calls of one observation 120 s together (`time_budget_s`),
and it never stops the reduction.
When a source fails, it writes `conditions/coverage_incomplete.json` instead of
`conditions/coverage.json`, and the next run of the reduction tries again.
It logs in to the ESO archive only with a password from the keyring or one entered
earlier in the same run, see [Reduce proprietary data](../how-to/proprietary-data.md).

The downloaded files are cached next to the raw data, in
`{raw_directory}/{IFS|IRDIS}/science/{target}/{filter}/{date}/SPARTA/`.
Set `config.conditions.refresh_cache = True` to query the archive again.
To switch the step off, set `config.steps.observing_conditions = False`.
The star amplitude is then 1 in every frame, with the status
`star_amplitude_no_wfs`.

The step writes `conditions/conditions.png`, which shows the frames and the
conditions on one time axis.

```{figure} images/hd95086_dbk12_2021-03-10_conditions.png
:alt: Eight panels of observing conditions against time for HD 95086, IRDIS DB_K12, 2021-03-11 UT.
:width: 100%

`conditions.png` of HD 95086, IRDIS `DB_K12`, night of 2021-03-10.
Shading marks the CORO (blue), CENTER (green) and FLUX (orange) frames, and dashed
lines mark a new SPARTA segment.
From the top, the panels show the WFS, infrared-loop and DTTS flux divided by their
medians, the SPARTA Strehl ratio, the seeing from SPARTA and MASS-DIMM, the coherence
time, the wind speed, the sky temperature and precipitable water vapour from LHATPRO,
and the DIMM flux scatter.
```

## FLUX blocks and which ones are usable

A {term}`FLUX block` is one group of consecutive FLUX frames, usually one before and
one after the coronagraphic sequence.
`calibrate_flux_psf` builds one PSF per block, after the FLUX frame vetting has
removed frames that do not show the star.

For each block and channel, `calibrate_star_amplitude` takes the median aperture flux
of the kept FLUX frames and the median WFS flux during them.
A block is not usable when, in any channel, its WFS flux is below half that of the
brightest block, or its ratio of aperture flux to WFS flux is below 70% of the
highest ratio.
Both tests catch only gross failures, such as a block taken through clouds or with
the star partly off the detector.
You can also leave blocks out by hand with `star_amplitude_exclude_blocks`.

Only the usable blocks enter the PSF that TRAP averages.
The `BLOCKS` extension of `star_amplitude.fits` records which ones they are.

## Stable or variable

The step calls a sequence stable when two numbers are small.
The level offset (`LEVOFF`) is the largest relative difference between the WFS flux
of a FLUX block and the median WFS flux of the science frames.
The drop depth (`DROPDEP`) is how far the faintest WFS sample falls below the 90th
percentile of all samples.

A sequence with a level offset of at most 3% and a drop depth of at most 10% is
stable and gets amplitude 1 in every frame.
Any other sequence is variable, and its amplitude follows the WFS flux.
In the 156 IRDIS sequences of a test sample reduced with this version, 28 were stable
and 111 variable.
The rest fell back to amplitude 1, for the reasons listed below.

## Anchoring the WFS flux to the FLUX frames

The WFS flux is in arbitrary units, so the step ties it to the FLUX frames, channel by
channel.
The scale is the median over the usable blocks of the ratio of aperture flux to WFS
flux.
The level is the mean aperture flux of the usable blocks, the flux of the PSF that
TRAP uses.
The amplitude of a science frame is

```{math}
A(t) = \frac{\mathrm{scale} \times \mathrm{WFS}(t)}{\mathrm{level}} ,
```

the aperture flux the star would have given in a FLUX frame at the WFS flux of that
science frame, divided by the flux of the PSF.

In HD 95086, IRDIS `DB_K12`, night of 2021-03-10, both FLUX blocks have 50 kept
frames.
Their WFS flux is 51.3 and 52.5, and their ratios in the first channel agree to
0.07%, so both are usable.

```text
channel,block,n_kept,flux,wfs,ratio,sharpness,psf_block,usable,excluded_by_hand
0,0,50,22749372.35081654,51.33638970588236,443143.20662502316,0.6589562746570932,0,True,False
0,1,50,23268194.621654384,52.524559997806264,442996.4691303687,0.6606426280415327,1,True,False
1,0,50,12325122.704542723,51.33638970588236,240085.49832109548,0.6272088534417162,0,True,False
1,1,50,12597423.734014759,52.524559997806264,239838.7294351614,0.6271059660738807,1,True,False
```

The first channel then has a scale of 443070 and a level of 23008783.
The first CORO frame has a WFS flux of 46.27, which gives
443070 × 46.27 / 23008783 = 0.891.
Over the 104 CORO frames the amplitude runs from 0.794 to 1.014, with a median of
0.931.
Without it, TRAP would have assumed amplitude 1 throughout and placed the companion
about 7% too faint.

How well the usable blocks agree on the ratio is recorded as `DISAGREE`, the median
over channels of the largest ratio divided by the smallest, minus one.
Above 0.15 for IRDIS and 0.25 for IFS the log gives a warning with the status
`star_amplitude_blocks_disagree`.
`SCALEUNC`, half of `DISAGREE`, is a rough uncertainty of the anchor.

## Gaps, the WFS floor and configuration changes

SPARTA records the WFS flux every 30 seconds in most sequences, and each reading is
one row of `conditions/sparta_visloop.csv`.
A frame shorter than that shares its reading with its neighbours, so the gap fill and
the check of configuration changes below work on the readings themselves rather than
on the frames.
SPARTA repeats the last reading of one of its files as the first of the next, so a
reading with the same value as the one before it counts once.
Without that table the step uses the samples of the science frames, counting a
repeated value as one reading, and logs the status
`star_amplitude_no_sparta_readings`.

A science frame without a WFS sample is filled from the readings next to its gap.
At each edge of the gap the step takes the median of up to
`star_amplitude_fill_samples` (5) readings on that side, using only readings within
`star_amplitude_fill_window_s` (300 s) of the reading next to the gap.
The gap is then filled by linear interpolation in time between the two edge values,
and at the start or end of the sequence with the one edge value there.
Only readings taken between the first and the last science frame enter the fill.
A single outlying reading next to a gap therefore does not set the whole fill.
In 221 sequences of the reduction of the spectral library (189 distinct series of
readings, because IFS and IRDIS of one observation share theirs), the difference
between two readings one minute apart has a robust standard deviation of 2.7%, four
minutes apart of 3.5%, and beyond about 16 minutes of 4.4%.
Readings were left out and predicted from their neighbours to compare fill rules.
On the data as recorded, the median of 3 to 8 readings within 300 seconds, or their
clipped mean within 150 to 600 seconds, predicts them as well as the single nearest
reading on each side.
When 5% of the readings were made 20% too high or too low, the error of the single
nearest readings rose from 2.9% to 4.4%, while that of the median of five stayed at
3.0% ([#311](https://github.com/m-samland/spherical/issues/311)).
`FILLNMIN` records the fewest readings behind any edge value, or the fewest frames
where a side of a gap has no readings.
When more than a quarter of the science frames are filled, the log gives a warning
with the status `star_amplitude_long_fill`.
When half or more of the frames of a variable sequence are filled, the amplitude is 1
with the status `star_amplitude_mostly_filled`.

A measured frame whose WFS flux is below `star_amplitude_wfs_floor` (0.1) times the
90th percentile of the sequence gets amplitude NaN, and TRAP treats it as a bad frame.
Filled frames are never set to NaN.

The WFS flux depends on the WFS mode, the spatial filter and the WFS filter, so the
step groups the SPARTA segments by these three into WFS configurations.
An ND filter change also starts a segment, but it does not reach the WFS, so such
segments count as the same configuration.
A different spatial filter can pass a different fraction of the light to the WFS.
The step measures this as the ratio of the WFS flux right after a change to that right
before it.
Placed at fake changes inside sequences with a single configuration, the same
estimate scatters by 2 to 4%, more across longer gaps without readings.
At the four changes during science frames in the spectral library the ratios are
0.94 to 1.01, so no step is detected beyond that scatter, but a real step of a few
percent cannot be excluded
([#311](https://github.com/m-samland/spherical/issues/311)).
A correction measured this way would add 2 to 4% of noise to every frame after the
change, so the step uses the WFS flux of every configuration that holds science
frames as measured, for the science frames and for the FLUX blocks.
A real step of a few percent then stays in the amplitude of the frames on one side
of the change, relative to the FLUX blocks on the other side.
A FLUX block in a configuration in which no science frame was taken gets no WFS level
and stays out of the PSF, because nothing ties its WFS flux to that of the science
frames.
For a sequence with a single configuration nothing changes.

At each change from one configuration of the science frames directly to another, the
step measures the ratio of the WFS flux after the change to that before it, from the
median edge values as for a gap, and logs it with the status `star_amplitude_wfs_config_change`
and the time without readings across the change.
`WFSSTEP` is the ratio farthest from 1.
When a ratio is more than 10% from 1, the log gives a warning with the status
`star_amplitude_wfs_step`, because the amplitude then rests on WFS flux that may not
be comparable across the change.
The estimate has wide tails, so between 5% and 11% of changes without any real step
also exceed 10%, more often across longer gaps without readings, and the warning
asks for a look at the WFS flux rather than marking a failure.
Changes to or from a configuration without science frames, as during the
acquisition, are not reported.
`SEGMENT` is the SPARTA segment that holds most science frames.

A correction factor could in principle also come from the FLUX blocks of each
configuration.
The step does not use them, because their FLUX over WFS ratio also changes with the
sharpness of the PSF.
The infrared flux within three pixels of the star drops when the seeing degrades,
while the visible WFS flux barely changes
([#320](https://github.com/m-samland/spherical/issues/320)).

## Continuous-waffle sequences

In a {term}`Continuous waffle` sequence the science frames are the CENTER frames.
By default (`star_amplitude_waffle_mode = "wfs"`) their amplitude follows the WFS flux
like in any other sequence.

With `star_amplitude_waffle_mode = "spots"`, the amplitude takes its shape from the
{term}`Waffle spots` instead, through `spot_amplitude_variation.fits`.
Its level comes from the WFS amplitude of the five science frames nearest to each
usable block.
The `CHANNELS` table then also has `spot_level_scatter`, how much those levels differ
between blocks.
Without a usable spot trace the amplitude follows the WFS, with the status
`star_amplitude_no_spot_trace`.

## When the step falls back to amplitude 1

In these cases every frame gets amplitude 1 and the log gives a warning with the
status.
TRAP then receives no star amplitude, but the `BLOCKS` mask still selects the PSF
blocks.

| Status | Cause | `BLOCKS` |
|---|---|---|
| `star_amplitude_no_vetting` | the FLUX frame vetting is off or its table is missing | empty, TRAP uses every block |
| `star_amplitude_no_wfs` | no matched condition tables, or no WFS sample during the science frames or during FLUX blocks in the configurations of the science frames | the blocks kept by the vetting |
| `star_amplitude_frame_mismatch` | the matched tables do not have the rows of the frame tables, force `observing_conditions` | the blocks kept by the vetting |
| `star_amplitude_no_usable_block` | no FLUX block passes the usability tests | the blocks kept by the vetting |
| `star_amplitude_mostly_filled` | half or more of the science frames of a variable sequence have no WFS sample | the usable blocks |

Only `star_amplitude_mostly_filled` keeps the anchor in `CHANNELS`, since the step
computed it before it gave up.
The statuses `star_amplitude_long_fill`, `star_amplitude_below_floor`,
`star_amplitude_block_left_out`, `star_amplitude_blocks_disagree`,
`star_amplitude_wfs_step` and `star_amplitude_no_spot_trace` only give a warning and leave the amplitude as computed.
`star_amplitude_wfs_config_change`, `star_amplitude_no_sparta_config` and
`star_amplitude_no_sparta_readings` are information messages.
Without a SPARTA configuration table the step treats the WFS samples of every segment
as one configuration.

## What TRAP receives

With `use_star_amplitude = True`, the default, `run_trap` passes the amplitude to
TRAP as the amplitude modulation of the forward model, and builds the PSF as the mean
over the blocks in `BLOCKS`.
Frames with amplitude NaN in any channel become bad frames, and the log gives a
warning when more than half of the frames are bad for this reason.
The star amplitude takes precedence over `pass_amplitude_modulation_to_trap`, the
older waffle-spot trace, which TRAP then ignores.
Set `use_star_amplitude = False` to give TRAP neither the amplitude nor the block
mask, so that it averages every PSF block.

```{expected-result}
`star_amplitude.fits` of HD 95086, IRDIS `DB_K12`, night of 2021-03-10.
The image has shape (2, 104), two channels and 104 CORO frames.

    MODE    = 'wfs     '
    REASON  = 'variable'
    LEVOFF  =  0.08664205415007187
    DROPDEP =  0.19148095527965603
    DISAGREE= 0.000680066794050193
    SCALEUNC= 0.000340033397025096
    NFLOOR  =                    0
    NFILLED =                    0
    MAXGAP_S=                  0.0
    FILLNMIN=
    SEGMENT =                    2
    NWFSCFG =                    1
    WFSSTEP =
    WFSFLOOR=                  0.1
    WAFMODE = 'wfs     '

The log line of the step reads
`Star amplitude mode wfs: sequence variable, block disagreement 0.001.`
```

The file and the block table are described on
[Output products](products.md#the-star-amplitude).

## Running the steps again

Both steps read only tables and small products, so they also run on observations
whose cubes were removed by cleanup.
To refresh the conditions and the amplitude in one run, force both, then run TRAP
again.

```python
config.steps = config.steps.merge(force={"observing_conditions", "calibrate_star_amplitude"})
```

```{common-mistake}
Forcing `observing_conditions` alone does not update `star_amplitude.fits`, because
both steps are {term}`Leaf step`s and a forced leaf re-runs only itself. Force
`calibrate_star_amplitude` too, and re-run TRAP, which reads the file.
```
