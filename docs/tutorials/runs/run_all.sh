#!/usr/bin/env bash
# Run every reduction the tutorials and the requirements page need, one after the other.
#
#   export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
#   export SPECIES_DIR=/path/to/species        # folder with species_database.hdf5
#   export NCPU=48
#   bash docs/tutorials/runs/run_all.sh
#
# Start it inside screen or tmux; it takes many hours. It stops at the first failure,
# and every step writes its log to $SPHERICAL_TUTORIAL_DIR/logs/.
set -euo pipefail

: "${SPHERICAL_TUTORIAL_DIR:?set SPHERICAL_TUTORIAL_DIR}"
: "${SPECIES_DIR:?set SPECIES_DIR}"
: "${NCPU:?set NCPU}"

cd "$(dirname "$0")/../../.."
LOG="$SPHERICAL_TUTORIAL_DIR/logs"
CSV="$SPHERICAL_TUTORIAL_DIR/csv"
HALF=$(( NCPU / 2 ))
mkdir -p "$LOG"

step() {  # step <log name> <command...>: run, keep the log, note the machine load
    local name=$1; shift
    echo "== $(date -u +%FT%TZ) start $name (load: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime))" | tee -a "$LOG/steps.log"
    "$@" 2>&1 | tee "$LOG/$name.log"
    echo "== $(date -u +%FT%TZ) end $name (load: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime))" | tee -a "$LOG/steps.log"
}
measure() {  # measure <instrument> <label> <filter>
    python docs/tools/measure_requirements.py --instrument "$1" --label "$2" --filter "$3" \
        --target "*_51_Eri" --date 2015-09-24 --out-dir "$CSV"
}

echo "spherical $(git describe --tags --dirty), NCPU=$NCPU, $(nproc 2>/dev/null || sysctl -n hw.ncpu) cores" | tee "$LOG/steps.log"
if [ ! -d "$SPHERICAL_TUTORIAL_DIR/database" ]; then
    step sync_tables spherical-sync-tables --dest "$SPHERICAL_TUTORIAL_DIR/database" --instrument all
fi

step irdis python docs/tutorials/runs/51eri_irdis.py --ncpu "$NCPU" --species-dir "$SPECIES_DIR"
step measure_irdis measure irdis 51eri_irdis DB_K12

step irdis_half python docs/tutorials/runs/51eri_irdis.py --ncpu "$HALF" --label "51eri_irdis_ncpu$HALF" --no-trap
step measure_irdis_half measure irdis "51eri_irdis_ncpu$HALF" DB_K12

step ifs python docs/tutorials/runs/51eri_ifs.py --ncpu "$NCPU" --species-dir "$SPECIES_DIR"
step measure_ifs measure ifs 51eri_ifs OBS_H

step annulus_irdis python docs/tutorials/runs/annulus_timing.py --instrument irdis --ncpu "$NCPU" \
    --inner 31 --outer 43 --species-dir "$SPECIES_DIR"
step annulus_ifs python docs/tutorials/runs/annulus_timing.py --instrument ifs --ncpu "$NCPU" \
    --inner 50 --outer 72 --species-dir "$SPECIES_DIR"

echo "ALL DONE" | tee -a "$LOG/steps.log"
