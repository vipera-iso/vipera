#!/usr/bin/env python3
"""Build docs/assets/favicon.ico from the rasterized logo.

Usage: python tools/make_favicon.py
Needs Pillow (dev dependency only; the CLI does not import it).
"""

from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
SIZES = [(16, 16), (32, 32), (48, 48), (64, 64)]


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed: python -m pip install pillow")
        return 1

    source = ASSETS / "logo.png"
    if not source.is_file():
        print(f"missing {source}; run `make assets-vectors` first")
        return 1

    with Image.open(source) as img:
        img.convert("RGBA").save(ASSETS / "favicon.ico", sizes=SIZES)

    print(f"wrote {ASSETS / 'favicon.ico'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
