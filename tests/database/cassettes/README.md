# Recorded ESO archive traffic

These cassettes let `test_resume.py` exercise the resume/incremental logic in the default
offline suite. Before them the tests cost ~15 minutes against the live archive, so they
were deselected by default — which meant the resume logic was covered by nothing anyone
actually ran, and its failures were as likely to be rate-limiting as regressions.

Recorded with [`pytest-recording`](https://github.com/kiwicom/pytest-recording) (a thin
plugin over `vcrpy`). Matching and header-filtering are configured by the `vcr_config`
fixture in `../conftest.py`.

## Why the window is what it is

`make_file_table` retrieves headers with **one HTTP GET per DP.ID**
(`archive.eso.org/hdr?DpId=...`), so the number of files in the date range *is* the cost —
of the tests, and of these files.

The night of **2020-03-07** holds 6 IFS files of a single target (`A124420`); extending to
2020-03-09 adds 2 `WAVE,LAMP` calibrations. That is 6 and 8 requests, against 171 and 305
for the 2016-09-15 window this replaced. Six files also halve cleanly for
`test_resume_with_simulated_partial_file`.

**ESO's `stime`/`etime` are night-based, not calendar-based.** `[2020-03-07, 2020-03-08]`
selects the night beginning 2020-03-07, whose exposures carry `2020-03-08` `DATE_OBS`
values. Verified against the live archive: `[2016-09-15, 2016-09-17]` returns 305 files,
which is `NIGHT_START` 2016-09-15 (171) + 2016-09-16 (134), not any calendar-date sum.
Keep this in mind when changing the window — picking dates off `DATE_OBS` will select the
wrong night.

## Re-recording

Needed when the ESO query interface changes, when `make_file_table` changes which requests
it issues, or when `test_live_archive_still_responds` (the live canary, `-m remote_data`)
starts failing.

> ### A partially-recorded cassette is silently green
>
> This is the trap, and it cost two wasted recordings here. `query_eso_data` catches
> request failures, logs an ERROR, and returns an empty list. So if the archive resets a
> connection mid-recording, the run still "succeeds" — it just writes a cassette with
> interactions missing. On replay those tests then pass against empty results, because
> most of them only assert `>= n_first` or compare two tables that are both empty.
>
> Two consequences:
>
> 1. **Nothing else may touch the ESO archive while recording.** Concurrent queries from
>    another shell are enough to trigger the resets. Record with the archive to yourself.
> 2. **Verify interaction counts afterwards — do not trust a green run.** Every broken
>    recording here still reported five passing tests; one of them "passed" in 18.78s with
>    cassettes that could never have worked on another machine. Counting is the only check
>    that caught it.
>
>    | test | form GET | wdb queries | header GETs | total |
>    |---|---|---|---|---|
>    | `test_fresh_run_creates_output_no_partial` | 2 | 2 | 6 | 10 |
>    | `test_incremental_extends_date_range` | 4 | 5 | 8 | 17 |
>    | `test_resume_with_simulated_partial_file` | 4 | 4 | 8 | 16 |
>    | `test_resume_false_ignores_partial` | 2 | 2 | 6 | 10 |
>    | `test_idempotent_rerun` | 4 | 4 | 6 | 14 |
>
>    The `form` GETs are astroquery fetching the WDB query form; there is one per query
>    because `cache=False` now reaches `query_instrument` too. Treat this table as a
>    tripwire, not a specification — it is a record of what the current code does, so a
>    deviation means *either* a bad recording *or* a real change in request behaviour.
>    Both are worth stopping for. (`test_incremental_extends_date_range` records one query
>    more than its form count; that asymmetry has not been run to ground.)

```sh
# delete the stale cassettes first — `once` will not overwrite an existing file
rm -rf tests/database/cassettes/test_resume/
pixi run -e dev pytest tests/database/test_resume.py -m "not remote_data" --record-mode=once

# then check completeness against the table above (counts requests, not matching lines)
python - <<'EOF'
import collections, pathlib, yaml
for f in sorted(pathlib.Path("tests/database/cassettes/test_resume").glob("*.yaml")):
    c = collections.Counter()
    for i in yaml.safe_load(f.read_text())["interactions"]:
        u = i["request"]["uri"]
        c["hdr" if "/hdr?" in u else "query" if u.endswith("/query") else "form"] += 1
    print(f"{f.stem:60s} {dict(c)}")
EOF
```

Prove the replay is offline rather than assuming it — both of these must hold:

```sh
# 1. replaying must not modify the cassettes (no "Appending request" / no mtime change)
ls -l tests/database/cassettes/test_resume/
pixi run -e dev pytest tests/database/test_resume.py -q
ls -l tests/database/cassettes/test_resume/

# 2. hiding a cassette must make its test FAIL — if it still passes, it is going live
```

## Two settings that look wrong and are not

**`record_mode` is not set in `vcr_config`.** pytest-recording already defaults to `none`
(an unmatched request is an error, never a silent live call), and pinning it in that dict
overrides `--record-mode` and makes re-recording impossible — with a confusing
"Can't overwrite existing cassette" error naming a file that does not exist.

**`body` is not in `match_on`.** astroquery 0.4.10 submits the WDB query as a multipart
POST whose boundary is regenerated per request (`--ab8319b16558684f024ba6ce6993d06a`), so
the body never compares equal between recording and replay. Matching on it fails every
query, and `query_eso_data` swallows the error, so the tests run green against empty
tables. The trade-off is that the two POSTs per batch (calibration, then science) share a
URL and are told apart only by recorded order — deterministic here, but it means a change
in *what* is asked for will not surface as a match failure.

**The `_isolate_astroquery_cache` fixture in `../conftest.py` is load-bearing.** astroquery
caches query responses on disk, and `make_file_table`'s `cache` argument does not reach
`query_instrument`. With a warm developer cache the query never becomes an HTTP request, so
the cassette records the header fetches but none of the queries that produce those DP.IDs —
replayable only on the machine that recorded it. Note the fixture patches `type(Eso)`, not
the imported `Eso`: astroquery exports a module-level *instance*, `cache_location` is a
property on `BaseQuery`, and `file_table` builds its own object with `Eso()`.

## What is in them

Public ESO archive metadata only — TAP query responses and FITS headers for public 2020
SPHERE data. `authorization`, `cookie` and `set-cookie` headers are filtered out at record
time. No credentials are involved; these queries are anonymous.
