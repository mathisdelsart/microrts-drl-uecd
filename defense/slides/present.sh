#!/bin/zsh
# Launch the defense deck in pdfpc with a 20-minute countdown timer.
#
#   ./present.sh           # normal launch (20 min countdown, warn last 2 min)
#   ./present.sh -s        # swap presenter / presentation screens
#   ./present.sh -P 30     # jump straight to page 30 (rehearse one part)
#
# Notes:
#   GST_REGISTRY_FORK=no avoids the GStreamer plugin-scanner hang on macOS.
#   Set the displays to "Extended" (not Mirror) so the big slide and the
#   presenter console land on different screens.
cd "$(dirname "$0")"
# Rebuild the notes from talk.md before every launch, so pdfpc never shows stale or
# shifted notes. We DELETE first, then regenerate from scratch: pdfpc may resave the
# sidecar with a different page count on quit, and patching that would misalign the
# tail. A fresh write always has exactly one note per PDF page, correctly ordered.
rm -f main.pdfpc
python3 ../script/build_pdfpc_notes.py >/dev/null 2>&1 || true
# -g           : one PDF page == one slide (so notes map by page number)
# -f markdown  : render the per-slide notes (main.pdfpc) as Markdown
# No -d/--duration: the timer is a count-up stopwatch from 0:00, so you can
# compare the elapsed time to the cumulative target shown at the end of each note.
GST_REGISTRY_FORK=no exec pdfpc -g -f markdown main.pdf "$@"
