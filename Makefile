PY ?= python3
ASSETS := docs/assets
RSVG ?= rsvg-convert
VHS ?= vhs
FFMPEG ?= ffmpeg

.PHONY: assets assets-vectors demo favicon verify clean-assets

## Regenerate every asset: rasters, favicon, demo GIF.
assets: assets-vectors favicon demo

## SVG -> PNG at the documented sizes (the SVGs are the source of truth).
assets-vectors:
	$(RSVG) -w 512 -h 512 $(ASSETS)/logo.svg -o $(ASSETS)/logo.png
	$(RSVG) -w 1024 -h 1024 $(ASSETS)/logo.svg -o $(ASSETS)/logo@2x.png
	$(RSVG) -w 1280 -h 320 $(ASSETS)/banner.svg -o $(ASSETS)/banner.png
	$(RSVG) -w 2560 -h 640 $(ASSETS)/banner.svg -o $(ASSETS)/banner@2x.png
	$(RSVG) -w 1280 -h 640 $(ASSETS)/social-preview.svg -o $(ASSETS)/social-preview.png
	@echo "vectors done"

## Multi-size .ico from logo.png
favicon:
	$(PY) tools/make_favicon.py

## Record docs/assets/demo.gif (needs vhs, ttyd, jq, figlet, ffmpeg, and a
## local 0.5B model). The tape records at 1800x1200; downscaling with lanczos
## is what keeps the text crisp once it is shown at 900px.
##
## A recording's length depends on how long the model takes to load, so the
## encoded size moves by ~10% between runs. Retry with fewer colours rather
## than let a fresh GIF trip check-added-large-files (limit 1000 KB).
demo:
	$(VHS) $(ASSETS)/demo.tape
	@set -e; for colours in 128 96; do \
	  $(FFMPEG) -v error -y -i $(ASSETS)/demo-raw.mp4 \
	    -vf "fps=12,scale=900:600:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=$$colours:stats_mode=full[p];[s1][p]paletteuse=dither=none" \
	    $(ASSETS)/demo.gif; \
	  size=$$(wc -c < $(ASSETS)/demo.gif); \
	  if [ $$size -le 972800 ]; then echo "demo.gif: $$size bytes ($$colours colours)"; break; fi; \
	  echo "demo.gif: $$size bytes with $$colours colours, retrying smaller"; \
	done
	@size=$$(wc -c < $(ASSETS)/demo.gif); \
	  test $$size -le 1024000 || { echo "demo.gif is $$size bytes (> 1000 KB); shorten the tape"; exit 1; }
	rm -f $(ASSETS)/demo-raw.mp4

## Everything CI checks.
verify:
	pre-commit run --all-files
	$(PY) -m pytest -q
	lychee --offline --include-fragments README.md docs
