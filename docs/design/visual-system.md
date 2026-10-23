---
title: Visual system
description: The tokens, type, and motion rules behind Vipera's terminal UI, README, and docs site.
---

# Visual system

Vipera does one thing that is worth designing around: it **allocates a
memory budget, layer by layer, and says no when the budget does not fit**. The
interface borrows its language from that act — a ledger. Numbers align, units
are always present, capacity is drawn rather than described, and nothing is
decorated unless it carries information.

This page is the source of truth for color, type, spacing, and motion across
the three surfaces: the Rich terminal UI (`src/vipera/ui.py`), the README, and
the docs landing page (`docs/index.html`).

## One identity, two contrast contexts

Before this pass the repository shipped two palettes: the SVG assets used
`#0B0F14` ink with a `#22D3EE` cyan accent, while the CLI used a slate blue
(`#5B7C99`) with near-white text (`#E4E8EC`). The banner and the terminal
looked like two different products, and the near-white `C_TEXT` was invisible
on a light terminal background.

The assets won: they are the identity people see first. The terminal now uses
the same ink / paper / cyan system, resolved through a theme so it stays
legible on both light and dark backgrounds.

## Tokens

| Token | Dark | Light | Used for |
|---|---|---|---|
| `ink` | `#0B0F14` | `#0B0F14` | text on light, page background on dark |
| `paper` | `#F1F5F9` | `#0B0F14` | primary text |
| `rule` | `#1E293B` | `#CBD5E1` | hairlines, table rules, empty gauge |
| `muted` | `#94A3B8` | `#52606D` | labels, units, footers |
| `accent` | `#22D3EE` | `#0E7490` | brand mark, focus, gauge fill, interactive prompt |
| `ok` | `#34D399` | `#047857` | a plan that loaded |
| `warn` | `#F5A524` | `#B45309` | a plan that was reduced |
| `error` | `#F87171` | `#B91C1C` | a load that failed |

Light-theme accent is cyan-700, not cyan-400: `#22D3EE` on white measures
1.81:1 and is unreadable as text; `#0E7490` measures 5.36:1. Every other pair
in the table clears 4.5:1 in both themes (muted text: 7.50:1 on ink, 6.46:1 on
white).

### Theme resolution

`ui.py` resolves the theme once at import, in this order:

1. `VIPERA_THEME=dark|light|none` — explicit override.
2. `NO_COLOR` set → `none`, which maps every token to Rich's `default` style so
   the CLI stays parseable in logs and pipes.
3. `COLORFGBG` with a background index of 7 or higher → `light`. Terminal
   emulators export this; anything else falls back to `dark`.

## Type

One family: whatever monospace the terminal already has. Vipera does not ship
a font, and a CLI that imposes one fights the user's setup.

The scale is defined by **role**, not size, since Rich has no font control:

| Role | Treatment | Example |
|---|---|---|
| Wordmark | accent, no bold, six-line block | `print_banner()` |
| Section | sentence case, muted, one hairline under it | `section("plan")` |
| Label | muted, lowercase, fixed column width | `ledger()` keys |
| Value | paper, right-aligned when numeric | `ledger()` values |
| Footer | muted, 2-space separated, no `·` | `214 tokens  6.2 tok/s  34.5 s` |

Headers are **sentence case**. The previous CLI used `SELECT A MODEL`,
`NODE / VERSION / RUNTIME` in caps; capitals cost horizontal space and shout,
and nothing in a terminal needs to shout.

Numbers are right-aligned and carry units at all times (`3.37 GB`, `14/40`,
`6.2 tok/s`). A bare `3.37` is a bug in a tool whose subject is units.

## Layout

```
  3  ▏ █████████████▏        14/40 layers      3.37 GB of 3.64 GB
    ▏ vr am ............   budget hit at layer 15
```

Terminal blocks are indented 3 spaces, because the brand mark is indented 3
and the eye should find one left edge. Column widths are set by the widest
label in a block, never by a guess.

The docs site uses a 71-character measure for prose and a full-bleed instrument
panel for the hero. Left aligned throughout; nothing is centered except the
single line that states what the tool is.

## Motion

One rule: **motion must show that something changed, and it must never delay
work**.

- The startup "heartbeat" sweep is gone. It ran about a second of animation
  before the first byte of useful output. The banner is now printed in a single
  frame.
- Token-by-token typing is gone from the default stream. The old code slept
  4–18 ms per character, which capped rendering near 90 characters/s no matter
  how fast the model was — roughly one token in four, so about 22 tok/s of
  English. Output now paints at a fixed 30 ms interval and nothing else.
- The progress bar keeps its fill animation: it represents a real wait, and the
  percentage is real.
- `prefers-reduced-motion` is respected on the docs site; the layer strip draws
  once and holds.

## What was removed, and why

| Removed | Reason |
|---|---|
| `#5B7C99` slate palette | Second identity; conflicted with the banner and logo |
| `C_TEXT = "bold #E4E8EC"` | 1.23:1 on a light terminal — the primary text was invisible |
| `·`-joined meta strings (`NODE · VERSION · RUNTIME`) | Read as decoration; replaced by an aligned ledger |
| ALL-CAPS table headers and titles | Cost width, gained nothing |
| Per-character typing delays | Made throughput-dependent rendering slower than the model |
| Rounded panels around every message | One border style for everything = no hierarchy |

The docs site is the one place with a real background and real type scale,
because a browser can honour them. The terminal does not pretend to.
