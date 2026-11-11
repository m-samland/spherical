#!/usr/bin/env bash
# Regenerate the arXiv preprint (paper.tex) from the JOSS sources (../paper.md, ../paper.bib)
# using the official JOSS build tool inara (https://github.com/openjournals/inara).
#
# References in the generated paper.tex are typeset by citeproc, so arXiv needs no .bib/.bst.
# Upload paper.tex with the figures (the tarball written below).
#
# Outputs (in this directory):
#   paper.tex                  arXiv source
#   paper.pdf                  xelatex build of paper.tex (arXiv's default engine)
#   paper_joss_reference.pdf   JOSS-styled PDF, for comparison only
#   arxiv_submission.tar.gz    upload bundle
#
# Requires docker. On Apple Silicon with colima, the amd64 image needs Rosetta
# (pandoc crashes under qemu):  colima start --vm-type vz --vz-rosetta
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOSS_DIR="$(dirname "$HERE")"
IMAGE=openjournals/inara

run() {
    # GHCRTS=-V0 switches off the timer of the Haskell runtime in pandoc. Under Rosetta its poll gets
    # interrupted and pandoc dies with "Ticker: poll failed: Interrupted system call".
    docker run --rm --platform linux/amd64 -u "$(id -u):$(id -g)" -e GHCRTS=-V0 "$@"
}

# inara writes its outputs next to paper.md
run -v "$JOSS_DIR:/data" "$IMAGE" -o preprint,pdf paper.md
mv "$JOSS_DIR/paper.preprint.tex" "$HERE/paper.tex"
mv "$JOSS_DIR/paper.pdf" "$HERE/paper_joss_reference.pdf"

# The preprint template is plain article class, whose default text block is 4.8 in wide and would shrink
# full-width figures to about 75%. One-inch margins give a 6.5 in text block.
perl -0pi -e 's/\]\{article\}\n/]{article}\n\\usepackage[margin=1in]{geometry}\n/ or die "documentclass line not found\n"' \
    "$HERE/paper.tex"

# paper.tex refers to the figures by file name, so they must sit next to it and go into the upload.
FIGURES=(overview_figure.png)
for figure in "${FIGURES[@]}"; do cp "$JOSS_DIR/$figure" "$HERE/"; done

# Check that the tex builds on a full TeX Live with arXiv's default engine (xelatex)
# and with pdflatex. The xelatex build is kept as paper.pdf.
for engine in pdflatex xelatex; do
    run -v "$HERE:/data" -w /data --entrypoint sh "$IMAGE" -c \
        "$engine -interaction=nonstopmode -halt-on-error paper.tex >/dev/null 2>&1 &&
         $engine -interaction=nonstopmode -halt-on-error paper.tex >/dev/null 2>&1" || true
    if [ ! -f "$HERE/paper.pdf" ] || grep -qE "undefined|^!" "$HERE/paper.log"; then
        grep -nE "undefined|^!" "$HERE/paper.log" || true
        echo "$engine reported problems, see paper.log" >&2
        exit 1
    fi
    [ "$engine" = xelatex ] || rm -f "$HERE/paper.pdf"
done
rm -f "$HERE"/paper.{aux,log,out}

tar -czf "$HERE/arxiv_submission.tar.gz" -C "$HERE" paper.tex "${FIGURES[@]}"
echo "Wrote paper.tex, paper.pdf, paper_joss_reference.pdf, arxiv_submission.tar.gz"
