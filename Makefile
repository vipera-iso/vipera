PY ?= python3
ASSETS := docs/assets
RSVG ?= rsvg-convert
VHS ?= vhs

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

## Record docs/assets/demo.gif (needs vhs, ttyd, jq, and a local 0.5B model).
demo:
	$(VHS) $(ASSETS)/demo.tape

## Everything CI checks.
verify:
	pre-commit run --all-files
	$(PY) -m pytest -q
	lychee --offline --include-fragments README.md docs
