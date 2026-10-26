#!/usr/bin/env bash
# Regenerate the arXiv preprint (paper.tex) from the JOSS sources (../paper.md, ../paper.bib)
# using the official JOSS build tool inara (https://github.com/openjournals/inara).
#
# The generated paper.tex is self-contained: references are typeset by citeproc,
# so arXiv needs no .bib/.bst. Upload paper.tex alone (or the tarball written below).
#
# Outputs (in this directory):
#   paper.tex                  arXiv source
#   paper.pdf                  pdflatex build of paper.tex (what arXiv will produce)
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
    docker run --rm --platform linux/amd64 -u "$(id -u):$(id -g)" "$@"
}

# inara writes its outputs next to paper.md
run -v "$JOSS_DIR:/data" "$IMAGE" -o preprint,pdf paper.md
mv "$JOSS_DIR/paper.preprint.tex" "$HERE/paper.tex"
mv "$JOSS_DIR/paper.pdf" "$HERE/paper_joss_reference.pdf"

# Check that the tex builds with pdflatex (arXiv's default) on a full TeX Live
run -v "$HERE:/data" -w /data --entrypoint sh "$IMAGE" -c \
    'pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null &&
     pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null'
if grep -qE "undefined|^!" "$HERE/paper.log"; then
    grep -nE "undefined|^!" "$HERE/paper.log"
    echo "pdflatex reported problems, see paper.log" >&2
    exit 1
fi
rm -f "$HERE"/paper.{aux,log,out}

tar -czf "$HERE/arxiv_submission.tar.gz" -C "$HERE" paper.tex
echo "Wrote paper.tex, paper.pdf, paper_joss_reference.pdf, arxiv_submission.tar.gz"
