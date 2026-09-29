#!/usr/bin/env bash
# Owner: V-NBM-infra. Render the NodeBasedModels.jl vignettes under the strict-cache policy (vignette rule 8) and
# fail on any error, on any warning, and on any missing or stale committed NetworkOutbreaks summary.
#
#   _shared/render.sh                         # every page (index.qmd and N*_*/index.qmd)
#   _shared/render.sh N01_foo/index.qmd       # only these pages (paths relative to vignettes/)
#
# The formats of _quarto.yml are rendered in separate passes: html, then pdf, then gfm LAST. A single
# `quarto render` of all three deletes each page's index_files/ once the self-contained (embed-resources) html is
# written, and with it the figures that the gfm index.md links to (index_files/figure-commonmark/*). Rendering
# gfm last keeps them. `check_pages.jl --rendered` then fails if any image of an index.md points to a missing file.
#
# Set NBM_VIGNETTES_NO_PDF=1 to skip the pdf pass where LaTeX is not installed (the pdf is then not refreshed).
# Run from anywhere; the log is written to _shared/.render.log (git-ignored by the vignette .gitignore's *.log).
set -euo pipefail
cd "$(dirname "$0")/.."
for a in "$@"; do
    case "$a" in
        -*) echo "render.sh: takes page paths only (got option $a); the formats come from _quarto.yml" >&2
            exit 2 ;;
    esac
done
export NETEPI_STRICT_CACHE=1
export JULIA_PKG_OFFLINE="${JULIA_PKG_OFFLINE:-true}"
log=_shared/.render.log
: > "$log"
julia --project=. _shared/check_pages.jl --self-test   # the checker's own parsers first (fails fast)
formats=(html)
[ "${NBM_VIGNETTES_NO_PDF:-0}" = "1" ] || formats+=(pdf)
formats+=(gfm)                                   # must stay last (see above)
for fmt in "${formats[@]}"; do
    echo "render.sh: quarto render --to $fmt $*" | tee -a "$log"
    status=0
    quarto render "$@" --to "$fmt" 2>&1 | tee -a "$log" || status=$?
    if [ "$status" -ne 0 ]; then
        echo "render.sh: quarto render --to $fmt failed (status $status); see $log" >&2
        exit "$status"
    fi
done
if grep -nE "WARN|Warning:|no valid committed summary|NETEPI_STRICT_CACHE is set|ERROR|LoadError" "$log"; then
    echo "render.sh: the render log above contains warnings or errors (rule 8)" >&2
    exit 1
fi
if [ "$#" -gt 0 ]; then
    ids=()
    for p in "$@"; do
        d=$(basename "$(dirname "$p")")
        case "$d" in N[0-9][0-9]_*) ids+=("${d%%_*}") ;; esac
    done
    [ "${#ids[@]}" -eq 0 ] && exit 0            # only index.qmd was rendered
    julia --project=. _shared/check_pages.jl --rendered "${ids[@]}"
else
    julia --project=. _shared/check_pages.jl --rendered
fi
