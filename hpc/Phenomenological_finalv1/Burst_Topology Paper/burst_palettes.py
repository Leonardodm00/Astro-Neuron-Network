#!/usr/bin/env python3
"""
burst_palettes.py -- poster color palettes + A0 sizing for the burst gallery.

PURE STYLE LAYER. No plotting, no file I/O, no matplotlib import. Everything here
is plain data (hex strings, point sizes, line widths) so it is trivially
unit-testable and so a restyle never touches detection (burst_metrics) or
rendering (burst_poster_plots). The render module consumes a PosterStyle and
asks it for an rcParams dict and figure sizes; it never hard-codes a color.

------------------------------------------------------------------------------
WHY THESE PALETTES
------------------------------------------------------------------------------
The poster's anchor color is BRAND = "#00363A": a very dark teal. In HSL its hue
is ~184 deg (cyan-teal), lightness ~11%, so on its own it is too dark to carry a
data figure; it is used here as the STRUCTURAL color (axes, spines, text, the
raster ink) that ties every figure to the poster, while a brighter ACCENT color
carries the instantaneous-firing-rate (IFR) envelope.

Three schemes are provided so the author can pick the one that best balances
on-brand harmony against feature legibility (the hero figure has only two ink
elements -- the black raster and one IFR fill -- so the accent hue carries it,
whereas the multi-detector analytical panels need three separable hues):

  "teal_analogous"     accent = brighter teal/cyan + an analogous blue.
                       Closest to the attached example (teal IFR). Most on-brand,
                       lowest accent-vs-paper contrast.
  "teal_amber_comp"    accent = amber/orange (the ~complement of 184 deg, ~4 deg)
                       with teal as the second pole. Journal-standard teal+amber:
                       highest pop, reasonably CVD-safe.
  "teal_triadic"       teal + amber + magenta (~64 deg / ~304 deg triad). For the
                       panels that must separate the TWO detectors + threshold.

The "threshold/alert" hue is deliberately kept warm and constant ACROSS schemes
(a semantic, not an aesthetic, choice: the same color always means "detection
threshold"); this is an intentional small departure from strict scheme purity
and is flagged here rather than hidden.

------------------------------------------------------------------------------
A0 SIZING
------------------------------------------------------------------------------
Figures are designed at a physical PANEL WIDTH (default 380 mm) so that when
placed at that size on an A0 sheet (841 x 1189 mm) the text is physically large
(axis labels ~30 pt, titles ~40 pt at scale 1.0 -- well above the ~24 pt rule of
thumb for A0 read at ~1.5 m). Point sizes adapt only GENTLY with panel width
(scale = (panel_mm / 380)^0.35) so a narrower or wider panel keeps roughly the
same on-poster physical text size without overflowing. PDF is emitted as well as
PNG: the PDF is vector, so it stays crisp at any final scale the layout demands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


BRAND = "#00363A"          # poster anchor: very dark teal (hue ~184 deg)
REF_PANEL_MM = 380.0       # reference panel width the point sizes are tuned for


# ===========================================================================
# Palette: one named set of role -> hex colors
# ===========================================================================
@dataclass(frozen=True)
class Palette:
    """Named role -> color map. Every figure reads colors by ROLE, never by hue,
    so swapping the palette restyles every figure identically."""
    name: str
    paper: str          # axes + figure background (near-white, faint teal tint)
    ink: str            # spines / ticks / axis + title text (the brand dark teal)
    muted: str          # secondary text / captions
    grid: str           # faint gridlines
    raster: str         # spike dots in the raster (reads black, teal-tinted)
    ifr_fill: str       # filled IFR envelope (the primary accent of the scheme)
    ifr_line: str       # IFR outline (a darker shade of ifr_fill)
    accent2: str        # second detector / secondary overlays
    threshold: str      # detection-threshold line (warm "alert" hue, scheme-const)
    span_pr: str        # population-rate detector burst-span shading
    span_li: str        # log-ISI / participation detector burst-span shading
    heat_lo: str        # activity-heatmap low end (== paper, for a clean floor)
    heat_hi: str        # activity-heatmap high end (saturated brand teal)

    def as_dict(self) -> Dict[str, str]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "name"}


# --- the three curated schemes (hexes hand-tuned for off-white at A0) --------
PALETTES: Dict[str, Palette] = {
    "teal_analogous": Palette(
        name="teal_analogous",
        paper="#F5F9F9", ink=BRAND, muted="#3F5E60", grid="#D7E3E3",
        raster="#0B2326",
        ifr_fill="#1AA0A8", ifr_line="#0C6E74",
        accent2="#2C7FB8",                         # analogous blue (~205 deg)
        threshold="#D2691E",                       # warm alert (semantic const)
        span_pr="#1AA0A8", span_li="#2C7FB8",
        heat_lo="#F5F9F9", heat_hi="#0E5A60",
    ),
    "teal_amber_comp": Palette(
        name="teal_amber_comp",
        paper="#F7F8F6", ink=BRAND, muted="#4A5A4E", grid="#E0E2DC",
        raster="#10211C",
        ifr_fill="#E08A1E", ifr_line="#A8610F",    # amber (~ complement, 4 deg)
        accent2="#0E7A82",                         # teal pole
        threshold="#B5452A",                       # deep warm alert
        span_pr="#E08A1E", span_li="#0E7A82",
        heat_lo="#F7F8F6", heat_hi="#0E5A60",
    ),
    "teal_triadic": Palette(
        name="teal_triadic",
        paper="#F5F8F8", ink=BRAND, muted="#3F5E60", grid="#D7E3E3",
        raster="#0B2326",
        ifr_fill="#1B9AA0", ifr_line="#0C6E74",    # teal
        accent2="#B5478B",                         # magenta triad pole (~304 deg)
        threshold="#C9761C",                       # amber triad pole (~64 deg)
        span_pr="#1B9AA0", span_li="#B5478B",
        heat_lo="#F5F8F8", heat_hi="#0E5A60",
    ),
}

DEFAULT_PALETTE_ORDER = ("teal_analogous", "teal_amber_comp", "teal_triadic")


# ===========================================================================
# PosterStyle: palette + physical size + derived point sizes / line widths
# ===========================================================================
@dataclass
class PosterStyle:
    """Everything a render function needs to draw at A0 panel size.

    panel_mm is the intended physical width of the figure ON the poster; the
    figure is created at exactly that width (in inches) so emitted point sizes
    are physical point sizes at that placement. dpi sets the PNG raster density
    (the companion PDF is always vector).
    """
    palette: Palette
    panel_mm: float = REF_PANEL_MM
    dpi: int = 300

    # --- derived (filled by __post_init__) ---
    scale: float = field(init=False)
    pt_base: float = field(init=False)
    pt_title: float = field(init=False)
    pt_label: float = field(init=False)
    pt_tick: float = field(init=False)
    pt_legend: float = field(init=False)
    pt_annot: float = field(init=False)     # metrics box (used in code)
    pt_caption: float = field(init=False)
    lw_spine: float = field(init=False)
    lw_grid: float = field(init=False)
    lw_ifr: float = field(init=False)       # IFR envelope line (used in code)
    lw_thresh: float = field(init=False)    # threshold line (used in code)
    lw_span_edge: float = field(init=False)
    marker_raster: float = field(init=False)  # scatter s (used in code)

    def __post_init__(self) -> None:
        # Gentle scaling: keep on-poster physical text size roughly constant
        # across panel widths (exponent < 1 damps the change).
        self.scale = (max(self.panel_mm, 1.0) / REF_PANEL_MM) ** 0.35
        s = self.scale
        self.pt_base = 22.0 * s
        self.pt_title = 40.0 * s
        self.pt_label = 30.0 * s
        self.pt_tick = 23.0 * s
        self.pt_legend = 23.0 * s
        self.pt_annot = 19.0 * s
        self.pt_caption = 18.0 * s
        self.lw_spine = 2.4 * s
        self.lw_grid = 1.0 * s
        self.lw_ifr = 3.2 * s
        self.lw_thresh = 2.6 * s
        self.lw_span_edge = 0.0
        self.marker_raster = 9.0 * s

    # ---- geometry -------------------------------------------------------
    def width_in(self) -> float:
        return self.panel_mm / 25.4

    def figsize(self, aspect_w_over_h: float) -> Tuple[float, float]:
        """(width_in, height_in) for a given width:height aspect ratio."""
        w = self.width_in()
        return (w, w / float(aspect_w_over_h))

    # ---- rcParams (generic styling only; role colors passed explicitly) -
    def rc(self) -> Dict[str, object]:
        """A plain dict for matplotlib rc_context. The render module FILTERS
        this against the installed matplotlib's valid keys before use, so this
        stays matplotlib-version- and import-free here."""
        p = self.palette
        return {
            "figure.facecolor": p.paper,
            "figure.edgecolor": p.paper,
            "axes.facecolor": p.paper,
            "savefig.facecolor": p.paper,
            "savefig.edgecolor": p.paper,
            "axes.edgecolor": p.ink,
            "axes.labelcolor": p.ink,
            "axes.titlecolor": p.ink,
            "axes.grid": False,
            "text.color": p.ink,
            "xtick.color": p.ink,
            "ytick.color": p.ink,
            "xtick.labelcolor": p.ink,
            "ytick.labelcolor": p.ink,
            "grid.color": p.grid,
            "grid.linewidth": self.lw_grid,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": self.pt_base,
            "axes.titlesize": self.pt_title,
            "axes.titleweight": "bold",
            "axes.labelsize": self.pt_label,
            "axes.labelweight": "medium",
            "xtick.labelsize": self.pt_tick,
            "ytick.labelsize": self.pt_tick,
            "legend.fontsize": self.pt_legend,
            "legend.frameon": False,
            "axes.linewidth": self.lw_spine,
            "xtick.major.width": self.lw_spine,
            "ytick.major.width": self.lw_spine,
            "xtick.major.size": 6.0 * self.scale,
            "ytick.major.size": 6.0 * self.scale,
            "lines.solidcapstyle": "round",
            "savefig.dpi": self.dpi,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # editable, embedded text in vector output (good for poster tweaks)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }


def make_style(palette_name: str = "teal_analogous",
               panel_mm: float = REF_PANEL_MM, dpi: int = 300) -> PosterStyle:
    """Factory: look up a palette by name and bundle it with a size spec."""
    if palette_name not in PALETTES:
        raise KeyError(f"unknown palette {palette_name!r}; "
                       f"choose from {sorted(PALETTES)}")
    return PosterStyle(palette=PALETTES[palette_name], panel_mm=panel_mm, dpi=dpi)


# ===========================================================================
# Tiny color utilities (no matplotlib) -- used by the smoke test and callers
# ===========================================================================
def hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    """'#RRGGBB' -> (r, g, b) in [0, 1]."""
    h = h.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #RRGGBB, got {h!r}")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def _rel_luminance(h: str) -> float:
    """WCAG relative luminance of a hex color in [0, 1]."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in hex_to_rgb01(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(h1: str, h2: str) -> float:
    """WCAG contrast ratio between two hex colors (1.0 .. 21.0)."""
    l1, l2 = _rel_luminance(h1), _rel_luminance(h2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def perceptual_distance(h1: str, h2: str) -> float:
    """Approximate perceptual color distance via the 'redmean' weighting.

    WCAG contrast is a LUMINANCE measure, so two colors of similar lightness but
    different hue (e.g. teal vs blue) score a low contrast yet are obviously
    distinguishable to a normal-vision viewer. For deciding whether two
    CATEGORICAL colors are separable we therefore use the redmean distance, a
    cheap closed-form approximation to CIE deltaE that accounts for hue:

        rbar = (r1 + r2) / 2
        d = sqrt((2 + rbar/256) dr^2 + 4 dg^2 + (2 + (255-rbar)/256) db^2)

    with channels in 0..255. The result lies in roughly [0, 765]; pairs above
    ~40 read as clearly different colors on a poster.
    """
    r1, g1, b1 = (c * 255.0 for c in hex_to_rgb01(h1))
    r2, g2, b2 = (c * 255.0 for c in hex_to_rgb01(h2))
    rbar = 0.5 * (r1 + r2)
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return float(((2 + rbar / 256.0) * dr * dr
                  + 4 * dg * dg
                  + (2 + (255 - rbar) / 256.0) * db * db) ** 0.5)


# ===========================================================================
# Smoke test  (python burst_palettes.py)
# ===========================================================================
def _smoke_test() -> int:
    print("Smoke test -- burst_palettes")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        ok = ok and bool(cond)

    roles = [f for f in Palette.__dataclass_fields__ if f != "name"]

    # 1. every palette defines every role with a parseable #RRGGBB hex
    for name, pal in PALETTES.items():
        d = pal.as_dict()
        missing = [r for r in roles if r not in d]
        check(f"{name}: all {len(roles)} roles present", not missing,
              f"missing={missing}" if missing else "")
        parse_ok = True
        for r, hx in d.items():
            try:
                hex_to_rgb01(hx)
            except Exception as e:
                parse_ok = False
                print(f"      bad hex for role {r}: {hx!r} ({e})")
        check(f"{name}: all hexes parse", parse_ok)

    # 2. legibility floors on off-white paper
    for name, pal in PALETTES.items():
        c_raster = contrast_ratio(pal.raster, pal.paper)
        c_ink = contrast_ratio(pal.ink, pal.paper)
        c_ifr = contrast_ratio(pal.ifr_fill, pal.paper)
        c_acc = contrast_ratio(pal.accent2, pal.paper)
        c_thr = contrast_ratio(pal.threshold, pal.paper)
        check(f"{name}: raster vs paper >= 7 (AAA text)", c_raster >= 7.0,
              f"ratio={c_raster:.1f}")
        check(f"{name}: ink vs paper >= 7", c_ink >= 7.0, f"ratio={c_ink:.1f}")
        check(f"{name}: ifr_fill vs paper >= 1.8 (fill, not text)", c_ifr >= 1.8,
              f"ratio={c_ifr:.2f}")
        check(f"{name}: accent2 vs paper >= 2.5", c_acc >= 2.5, f"ratio={c_acc:.2f}")
        check(f"{name}: threshold vs paper >= 2.5", c_thr >= 2.5, f"ratio={c_thr:.2f}")
        # the categorical hues must be PERCEPTUALLY distinct (separable panels);
        # use redmean distance, not WCAG contrast, which is luminance-only.
        d_pair = perceptual_distance(pal.ifr_fill, pal.accent2)
        check(f"{name}: ifr_fill vs accent2 perceptually distinct (>=40)",
              d_pair >= 40.0, f"redmean={d_pair:.0f}")

    # 2b. the triadic scheme exists to separate THREE categories at once, so all
    #     three of its categorical hues must be mutually distinct.
    tri = PALETTES["teal_triadic"]
    d_fa = perceptual_distance(tri.ifr_fill, tri.accent2)
    d_ft = perceptual_distance(tri.ifr_fill, tri.threshold)
    d_at = perceptual_distance(tri.accent2, tri.threshold)
    check("teal_triadic: ifr_fill/accent2/threshold mutually distinct (>=40)",
          min(d_fa, d_ft, d_at) >= 40.0,
          f"min redmean={min(d_fa, d_ft, d_at):.0f}")

    # 3. brand anchor used as ink everywhere
    check("BRAND is the ink in every palette",
          all(p.ink == BRAND for p in PALETTES.values()))

    # 4. PosterStyle scaling monotone in panel width, geometry sane
    s_narrow = make_style("teal_analogous", panel_mm=250)
    s_ref = make_style("teal_analogous", panel_mm=380)
    s_wide = make_style("teal_analogous", panel_mm=840)
    check("scale monotone increasing in panel_mm",
          s_narrow.scale < s_ref.scale < s_wide.scale,
          f"{s_narrow.scale:.2f} < {s_ref.scale:.2f} < {s_wide.scale:.2f}")
    check("reference panel -> scale == 1.0", abs(s_ref.scale - 1.0) < 1e-9,
          f"scale={s_ref.scale:.4f}")
    check("reference title size == 40 pt", abs(s_ref.pt_title - 40.0) < 1e-9,
          f"{s_ref.pt_title:.1f} pt")
    w, h = s_ref.figsize(2.2)
    check("figsize(2.2) width == 380 mm", abs(w * 25.4 - 380.0) < 1e-6,
          f"{w * 25.4:.1f} mm")
    check("figsize(2.2) aspect == 2.2", abs((w / h) - 2.2) < 1e-9,
          f"{w / h:.3f}")

    # 5. rc dict is a plain dict of expected keys (no matplotlib needed)
    rc = s_ref.rc()
    check("rc() returns dict with font.size & axes.edgecolor",
          isinstance(rc, dict) and "font.size" in rc and "axes.edgecolor" in rc)
    check("rc axes.edgecolor == BRAND", rc.get("axes.edgecolor") == BRAND)

    # 6. factory rejects unknown palette
    bad = False
    try:
        make_style("not_a_palette")
    except KeyError:
        bad = True
    check("make_style rejects unknown palette name", bad)

    print("\nSmoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
