#!/usr/bin/env bash
# Find out why 51 Eri b fails TRAP's shape check in the 2015 IRDIS tutorial run, and
# measure preprocessing times on an idle machine. See README.md next to this file.
#
#   export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
#   export SPECIES_DIR=/path/to/species        # folder with species_database.hdf5
#   export NCPU=48
#   bash docs/tutorials/runs/investigation/run_investigation.sh
#
# Needs the 51eri_irdis run of run_all.sh in $SPHERICAL_TUTORIAL_DIR. Start it inside
# screen or tmux; it takes about eight hours. A failed step is logged and the next one
# starts. An experiment whose experiment.json exists is skipped, so the script can be
# restarted after an interruption (remove the half-written experiment folder first).
set -uo pipefail

: "${SPHERICAL_TUTORIAL_DIR:?set SPHERICAL_TUTORIAL_DIR}"
: "${SPECIES_DIR:?set SPECIES_DIR}"
: "${NCPU:?set NCPU}"
RUN_IFS_TIMING=${RUN_IFS_TIMING:-1}

cd "$(dirname "$0")/../../../.."
T="$SPHERICAL_TUTORIAL_DIR"
LOG="$T/investigation_logs"
CSV="$T/investigation_csv"
EXP="$T/experiments"
INV=docs/tutorials/runs/investigation
mkdir -p "$LOG" "$CSV" "$EXP"

machine() {  # load average and usable cores; nproc honours OMP_NUM_THREADS, so ask Python
    python -c "import os; a = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(); print('load %.1f %.1f %.1f, %d usable cores' % (*os.getloadavg(), a))"
}
step() {  # step <log name> <command...>: run, keep the log, note load and exit status
    local name=$1; shift
    echo "== $(date -u +%FT%TZ) start $name ($(machine))" | tee -a "$LOG/steps.log"
    "$@" > "$LOG/$name.log" 2>&1
    local status=$?
    echo "== $(date -u +%FT%TZ) end $name exit=$status ($(machine))" | tee -a "$LOG/steps.log"
}
measure() {  # measure <instrument> <label> <filter> <date>
    python docs/tools/measure_requirements.py --instrument "$1" --label "$2" --filter "$3" \
        --target "*_51_Eri" --date "$4" --out-dir "$CSV"
}
exp() {  # exp <source> <name> <trap_experiment.py options...>
    local source=$1 name=$2; shift 2
    if [ -f "$EXP/$source/$name/experiment.json" ]; then
        echo "-- skip $source/$name (done)" | tee -a "$LOG/steps.log"; return
    fi
    step "exp_${source}_$name" python "$INV/trap_experiment.py" --source "$source" --name "$name" \
        --ncpu "$NCPU" --species-dir "$SPECIES_DIR" "$@"
}

echo "spherical $(git describe --tags --dirty), NCPU=$NCPU, $(machine)" | tee -a "$LOG/steps.log"
if [ ! -d "$T/51eri_irdis/reduction" ]; then
    echo "$T/51eri_irdis/reduction is missing; run run_all.sh first." | tee -a "$LOG/steps.log"; exit 1
fi

S=51eri_irdis
B15=-35.95,-8.43   # yx_known_companion_position of the 2015 regression run
B17=-33.40,-15.10  # Maire et al. 2019: 449.0 mas (36.65 px), PA 155.67 deg
S17=51eri_irdis_2017

# Fakes sit 72 deg apart and at least 72 deg from b, so no two tracks overlap during
# the 41.5 deg (2015) or 52.6 deg (2017) of field rotation.
fakes() {  # fakes <sep> <scale> <PA offsets from b...>: --inject options, 2015
    local sep=$1 scale=$2; shift 2
    for off in "$@"; do printf -- '--inject %s:%s:%s ' "$sep" "$(python -c "print(round((166.84 + $off) % 360, 2))")" "$scale"; done
}
fakes17() {  # the same for 2017
    local sep=$1 scale=$2; shift 2
    for off in "$@"; do printf -- '--inject %s:%s:%s ' "$sep" "$(python -c "print(round((155.67 + $off) % 360, 2))")" "$scale"; done
}
# --- 0. Smoke test: the brightest fakes must come back where they were injected ---------
# shellcheck disable=SC2046
exp $S inj37_s2_A_known --inner 31 --outer 43 --frac 0.20 --known=$B15 $(fakes 37 2 72 144 216 288)
mkdir -p "$EXP/analysis"
if ! python "$INV/analyse_experiments.py" "$EXP/$S/inj37_s2_A_known" --out "$EXP/analysis/smoke" \
        --require-fakes 0.75 > "$LOG/smoke.log" 2>&1; then
    echo "Smoke test failed: the injected planets were not recovered, see $LOG/smoke.log" | tee -a "$LOG/steps.log"
    exit 1
fi

# --- 1. Preprocessing time on the idle machine (IRDIS 2015, no TRAP) ---------------------
for n in "$NCPU" $((NCPU / 2)) $((NCPU / 4)) 4; do
    label="timing_irdis_ncpu$n"
    if [ ! -f "$T/$label/run_provenance.json" ]; then
        step "$label" python docs/tutorials/runs/51eri_irdis.py --ncpu "$n" --label "$label" --no-trap
        step "measure_$label" measure irdis "$label" DB_K12 2015-09-24
    fi
done

# --- 2. The 2017 sequence: preprocessing only, TRAP comes in the experiments --------------
if [ ! -f "$T/51eri_irdis_2017/run_provenance.json" ]; then
    step irdis_2017 python docs/tutorials/runs/51eri_irdis_2017.py --ncpu "$NCPU" --no-trap \
        --species-dir "$SPECIES_DIR"
    step measure_irdis_2017 measure irdis 51eri_irdis_2017 DB_K12 2017-09-27
fi


# --- 3. 2015: component fraction x known position, on the regression annulus -------------
for f in 0.10 0.15 0.20 0.25 0.30; do
    exp $S "a31-43_f${f}_blind" --inner 31 --outer 43 --frac $f
    exp $S "a31-43_f${f}_known" --inner 31 --outer 43 --frac $f --known=$B15
done

# --- 4. 2015: fake planets with b's spectrum (b masked as known), to measure how often ---
#        the shape check rejects a real planet.
for sep in 25 37 55; do
    inner=$((sep - 6)); outer=$((sep + 6))
    for scale in 1 2; do
        # shellcheck disable=SC2046
        exp $S "inj${sep}_s${scale}_A_known" --inner $inner --outer $outer --frac 0.20 --known=$B15 $(fakes $sep $scale 72 144 216 288)
        # shellcheck disable=SC2046
        exp $S "inj${sep}_s${scale}_B_known" --inner $inner --outer $outer --frac 0.20 --known=$B15 $(fakes $sep $scale 108 180 252)
    done
done
for scale in 1 2; do  # the same fakes with nothing masked: a fully blind search
    # shellcheck disable=SC2046
    exp $S "inj37_s${scale}_A_blind" --inner 31 --outer 43 --frac 0.20 $(fakes 37 $scale 72 144 216 288)
    # shellcheck disable=SC2046
    exp $S "inj37_s${scale}_B_blind" --inner 31 --outer 43 --frac 0.20 $(fakes 37 $scale 108 180 252)
done

# --- 5. 2017 (deeper data; ground truth for the shape of 51 Eri b) -----------------------
for f in 0.15 0.20; do
    exp $S17 "a31-43_f${f}_blind" --inner 31 --outer 43 --frac $f
    exp $S17 "a31-43_f${f}_known" --inner 31 --outer 43 --frac $f --known=$B17
done
for scale in 1 2; do
    # shellcheck disable=SC2046
    exp $S17 "inj37_s${scale}_A_known" --inner 31 --outer 43 --frac 0.20 --known=$B17 $(fakes17 37 $scale 72 144 216 288)
done

# --- 6. Search region: inner bound, outer bound, the tutorial candidate 1-100 px ----------
exp $S "r1-43_f0.20_known" --inner 1 --outer 43 --frac 0.20 --known=$B15
exp $S "r1-43_f0.20_blind" --inner 1 --outer 43 --frac 0.20
exp $S "r31-100_f0.20_known" --inner 31 --outer 100 --frac 0.20 --known=$B15
exp $S "r1-100_f0.20_known" --inner 1 --outer 100 --frac 0.20 --known=$B15
exp $S "r1-100_f0.20_blind" --inner 1 --outer 100 --frac 0.20
exp $S "r1-100_f0.15_blind" --inner 1 --outer 100 --frac 0.15
exp $S17 "r1-100_f0.20_known" --inner 1 --outer 100 --frac 0.20 --known=$B17
exp $S17 "r1-100_f0.20_blind" --inner 1 --outer 100 --frac 0.20
exp $S "r1-200_f0.20_known" --inner 1 --outer 200 --frac 0.20 --known=$B15

# --- 7. Analysis (small files; copy $EXP/analysis back) ----------------------------------
mkdir -p "$EXP/analysis"
step analysis python "$INV/analyse_experiments.py" "$EXP"/*/* \
    "$T/51eri_irdis/reduction/IRDIS/trap/*_51_Eri/DB_K12/2015-09-24" \
    --out "$EXP/analysis/experiments" --stamps "$EXP/analysis/stamps.png"

# --- 8. Preprocessing time on the idle machine (IFS, no TRAP; ~30 min and ~20 GB each) ----
if [ "$RUN_IFS_TIMING" = 1 ]; then
    for n in "$NCPU" $((NCPU / 2)); do
        label="timing_ifs_ncpu$n"
        if [ ! -f "$T/$label/run_provenance.json" ]; then
            step "$label" python docs/tutorials/runs/51eri_ifs.py --ncpu "$n" --label "$label" --no-trap
            step "measure_$label" measure ifs "$label" OBS_H 2015-09-24
        fi
    done
fi

echo "ALL DONE $(date -u +%FT%TZ)" | tee -a "$LOG/steps.log"
