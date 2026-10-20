# Brand assets

| File | Dimensions | Purpose |
|---|---|---|
| `logo.svg` / `logo.png` / `logo@2x.png` | 512×512, 512, 1024 | App icon, favicon source, README/OG fallback. |
| `banner.svg` / `banner.png` / `banner@2x.png` | 1280×320, 1280×320, 2560×640 | README header. |
| `social-preview.svg` / `social-preview.png` | 1280×640 | GitHub repository social preview. Must be uploaded manually (Settings → Social preview); the API cannot set it. |
| `favicon.ico` | 16, 32, 48, 64 | Multi-size icon for docs sites and browsers. |
| `demo.gif` | 900×600, 15 fps, ~14 s | README demo, recorded with `vhs`. |

## Palette

| Role | Hex |
|---|---|
| Background | `#0B0F14` |
| Accent (cyan) | `#22D3EE` |
| Foreground | `#F1F5F9` |
| Muted text | `#94A3B8` |
| Panel / border | `#111827` / `#1F2937` |

## Typography

- Wordmark and UI text: **Inter SemiBold**, falling back to `DejaVu Sans`.
- Terminal and code snippets: **JetBrains Mono**.

Inter is not bundled. Install it locally (`https://rsms.me/inter/`) before
regenerating if you want the exact wordmark; otherwise the rasterizer uses the
documented fallback and the layout still holds.

## Regenerate

```bash
make assets          # vectors, raster sizes, favicon, demo GIF
make assets-vectors  # SVG → PNG only (no terminal recording)
```

Requirements: `rsvg-convert` (librsvg), `vhs` + `ttyd` for the GIF, `jq` for the
demo request, and Pillow for the `.ico`. The demo tape serves
`qwen2.5-0.5b-instruct-q4_k_m` on port 8686; adjust `demo.tape` to your own
alias before recording.

`docs/assets/demo-request.sh` is the streaming client shown in the GIF. It
reads SSE frames and prints only the text deltas, so the recording shows tokens
rather than JSON.

Keep every file in this directory under 2 MB — CI enforces it
(`.github/workflows/assets.yml`).
