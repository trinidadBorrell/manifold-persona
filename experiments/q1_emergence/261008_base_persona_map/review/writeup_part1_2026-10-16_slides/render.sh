#!/bin/bash
# Render the deck to PDF and PNG without LibreOffice: macOS Quick Look turns the
# .pptx into HTML, headless Chrome prints that HTML to PDF, pdftoppm makes PNGs.
# This is Apple's Office renderer, so small layout differences from PowerPoint are possible.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DECK="$HERE/e3_checkin_2026-10-16.pptx"
WORK="${TMPDIR:-/tmp}/e3_deck_render"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

rm -rf "$WORK" && mkdir -p "$WORK"
qlmanage -p -o "$WORK" "$DECK" >/dev/null 2>&1 &
QL=$!
for _ in $(seq 1 30); do
  [ -f "$WORK/$(basename "$DECK").qlpreview/Preview.html" ] && break
  sleep 1
done
sleep 2
kill "$QL" 2>/dev/null || true

PREV="$WORK/$(basename "$DECK").qlpreview"
# One slide per page, no screen margins or shadows.
sed -i '' 's|</head>|<style>@page{size:959px 540px;margin:0}html,body{margin:0;padding:0;background:#fff}div.slide{margin:0!important;box-shadow:none!important;-webkit-box-shadow:none!important;page-break-after:always;break-after:page}</style></head>|' "$PREV/Preview.html"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --user-data-dir="$WORK/chrome" \
  --print-to-pdf="$HERE/e3_checkin_2026-10-16.pdf" "file://$PREV/Preview.html" >/dev/null 2>&1

rm -f "$HERE"/png/*.png
mkdir -p "$HERE/png"
pdftoppm -png -r 128 "$HERE/e3_checkin_2026-10-16.pdf" "$HERE/png/slide"
ls "$HERE/png"
