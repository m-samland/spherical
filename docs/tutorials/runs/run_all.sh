#!/usr/bin/env bash
# Run the two reduction tutorials (IRDIS, then IFS), each followed by its measurement.
#
#   export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
#   export SPECIES_DIR=$SPHERICAL_TUTORIAL_DIR/species   # private copy, see README.md
#   export NCPU=48
#   bash docs/tutorials/runs/run_all.sh
#
# Start it inside screen or tmux. It takes about an hour on 48 cores plus the download, and
# longer on a busy node. It stops at the first failure,
# and every step writes its log to $SPHERICAL_TUTORIAL_DIR/logs/.
set -euo pipefail

: "${SPHERICAL_TUTORIAL_DIR:?set SPHERICAL_TUTORIAL_DIR}"
: "${SPECIES_DIR:?set SPECIES_DIR}"
: "${NCPU:?set NCPU}"
if [ ! -f "$SPECIES_DIR/species_database.hdf5" ]; then
    echo "No species_database.hdf5 in SPECIES_DIR=$SPECIES_DIR" >&2; exit 1
fi

cd "$(dirname "$0")/../../.."
LOG="$SPHERICAL_TUTORIAL_DIR/logs"
CSV="$SPHERICAL_TUTORIAL_DIR/csv"
mkdir -p "$LOG"

machine() {  # load average and usable cores; nproc honours OMP_NUM_THREADS, so ask Python
    python -c "import os; a = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(); print('load %.1f %.1f %.1f, %d usable cores' % (*os.getloadavg(), a))"
}
step() {  # step <log name> <command...>: run, keep the log, note the machine load
    local name=$1; shift
    echo "== $(date -u +%FT%TZ) start $name ($(machine))" | tee -a "$LOG/steps.log"
    "$@" 2>&1 | tee "$LOG/$name.log"
    echo "== $(date -u +%FT%TZ) end $name ($(machine))" | tee -a "$LOG/steps.log"
}
measure() {  # measure <instrument> <label> <filter>
    python docs/tools/measure_requirements.py --instrument "$1" --label "$2" --filter "$3" \
        --target "*_51_Eri" --date 2015-09-24 --out-dir "$CSV"
}

echo "spherical $(git describe --tags --dirty), NCPU=$NCPU, $(machine)" | tee "$LOG/steps.log"
if [ ! -d "$SPHERICAL_TUTORIAL_DIR/database" ]; then
    step sync_tables spherical-sync-tables --dest "$SPHERICAL_TUTORIAL_DIR/database" --instrument all
fi

step irdis python docs/tutorials/runs/51eri_irdis.py --ncpu "$NCPU" --species-dir "$SPECIES_DIR"
step measure_irdis measure irdis 51eri_irdis DB_K12

step ifs python docs/tutorials/runs/51eri_ifs.py --ncpu "$NCPU" --species-dir "$SPECIES_DIR"
step measure_ifs measure ifs 51eri_ifs OBS_H

echo "ALL DONE" | tee -a "$LOG/steps.log"
