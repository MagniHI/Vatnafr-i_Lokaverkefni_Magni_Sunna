"""
Skref 7 – Leitnigreining (Trend Analysis)
==========================================
Notar:
  • Theil-Sen estimator  (Sen, 1968; Theil, 1950)  – magnmat á leitni
  • Modified Mann-Kendall test  (Hamed & Ramachandra Rao, 1998)
    → leiðréttir fyrir sjálffylgni í gögnum; hentugur fyrir vatnafræðileg gögn

Tölfræðileg marktækni metin við p < 0.05.

Greining:
  A) Árleg leitni – Q, P, T
  B) Mánaðarleg leitni – Q fyrir hvern mánuð (12 greiningartímabil)
  C) Árstíðaleitni – vor, sumar, haust, vetur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pymannkendall as mk
from scipy.stats import theilslopes
from pathlib import Path

DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# GÖGN
# ════════════════════════════════════════════════════════════════════════════
flow = pd.read_csv(DATA_DIR / "olfus_flaedi_93_23.csv", parse_dates=["date"])
flow = flow.set_index("date")
met  = pd.read_csv(DATA_DIR / "olfus_vedur_93_23.csv",  parse_dates=["date"])
met  = met.set_index("date")
df   = flow.join(met)

# Vatnafræðilegt ár (okt–sep) sem auðkenni
df["vy"] = np.where(df.index.month >= 10, df.index.year + 1, df.index.year)


# ════════════════════════════════════════════════════════════════════════════
# HJÁLPARFÖLL
# ════════════════════════════════════════════════════════════════════════════

def leitnigreining(y_series, label=""):
    """
    Keyrir modified Mann-Kendall (H&R 1998) og Theil-Sen á tímaröð.

    Skilar dict með:
      slope   – Theil-Sen hallatala [eining/ár]
      pct_chg – hlutfallsleg breyting yfir 30 ár [%]
      p       – p-gildi modified Mann-Kendall
      sig     – True ef p < 0.05
      trend   – 'aukast' / 'minnka' / 'engin'
    """
    y = y_series.dropna().values
    x = np.arange(len(y), dtype=float)

    # Modified Mann-Kendall (Hamed & Ramachandra Rao, 1998)
    res = mk.hamed_rao_modification_test(y)

    # Theil-Sen hallatala [eining á hverja tímabilseiningu (ár eða mánuð)]
    slope, intercept, _, _ = theilslopes(y, x)

    # Heildarbreyting yfir tímabilið [%]
    y_start = intercept
    y_end   = intercept + slope * (len(y) - 1)
    pct_chg = (y_end - y_start) / abs(y_start) * 100 if y_start != 0 else np.nan

    sig = res.p < 0.05
    if res.p < 0.05:
        trend = "aukast" if slope > 0 else "minnka"
    else:
        trend = "engin"

    if label:
        sig_str = "**MARKTÆK**" if sig else "ekki marktæk"
        print(f"  {label:<30}  slope={slope:+.4f}  p={res.p:.3f}  ({sig_str})  {trend}")

    return {
        "slope": slope, "intercept": intercept,
        "pct_chg": pct_chg, "p": res.p,
        "tau": res.Tau, "sig": sig, "trend": trend,
        "y": y, "x": x,
    }


# ════════════════════════════════════════════════════════════════════════════
# A) ÁRLEG LEITNI
# ════════════════════════════════════════════════════════════════════════════
print("\n══ A) ÁRLEG LEITNI ══")

ar_Q = df.groupby("vy")["Q_m3s"].mean()
ar_P = df.groupby("vy")["P_mm"].mean()
ar_T = df.groupby("vy")["T_degC"].mean()

r_Q = leitnigreining(ar_Q, "Q [m³/s] árlegt meðaltal")
r_P = leitnigreining(ar_P, "P [mm/dag] árlegt meðaltal")
r_T = leitnigreining(ar_T, "T [°C] árlegt meðaltal")


# ════════════════════════════════════════════════════════════════════════════
# B) MÁNAÐARLEG LEITNI Á Q
# ════════════════════════════════════════════════════════════════════════════
print("\n══ B) MÁNAÐARLEG LEITNI – Q ══")

manud_nofn = ["Jan","Feb","Mar","Apr","Maí","Jún",
              "Júl","Ágú","Sep","Okt","Nóv","Des"]

mo_results = {}
for m in range(1, 13):
    mask  = df.index.month == m
    # Nota vatnafræðilegt ár sem tímaás
    q_mo  = df.loc[mask, "Q_m3s"].groupby(df.loc[mask, "vy"]).mean()
    r     = leitnigreining(q_mo, f"Q – {manud_nofn[m-1]}")
    mo_results[m] = r


# ════════════════════════════════════════════════════════════════════════════
# C) ÁRSTÍÐALEITNI
# ════════════════════════════════════════════════════════════════════════════
print("\n══ C) ÁRSTÍÐALEITNI – Q ══")

arstimar = {
    "Vetur (Des–Feb)": [12, 1, 2],
    "Vor   (Mar–Maí)": [3,  4, 5],
    "Sumar (Jún–Ágú)": [6,  7, 8],
    "Haust (Sep–Nóv)": [9, 10, 11],
}
season_results = {}
for name, months in arstimar.items():
    mask = df.index.month.isin(months)
    q_s  = df.loc[mask, "Q_m3s"].groupby(df.loc[mask, "vy"]).mean()
    r    = leitnigreining(q_s, f"Q – {name}")
    season_results[name] = r


# ════════════════════════════════════════════════════════════════════════════
# MYNDIR
# ════════════════════════════════════════════════════════════════════════════

# ── Mynd 1: Árleg leitni – Q, P, T ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.suptitle("Árleg leitni – Ölfusá, Selfoss  (1993–2023)", fontsize=12, fontweight="bold")

plot_data = [
    (ar_Q, r_Q, "Rennsli Q [m³/s]",  "#2c7bb6", "Q"),
    (ar_P, r_P, "Úrkoma P [mm/dag]", "#1a9641", "P"),
    (ar_T, r_T, "Hitastig T [°C]",   "#d7191c", "T"),
]

for ax, (series, r, ylabel, clr, tag) in zip(axes, plot_data):
    yrs = series.index.values
    ax.plot(yrs, series.values, "o-", color=clr, ms=5, lw=1.4, alpha=0.8)

    # Theil-Sen leitni
    fit = r["intercept"] + r["slope"] * r["x"]
    ls  = "-" if r["sig"] else "--"
    lbl = (f"Theil-Sen: {r['slope']:+.3f}/ár  "
           f"(p={r['p']:.3f}{'  *' if r['sig'] else ''})")
    ax.plot(yrs, fit, ls, color="k", lw=1.8, label=lbl)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlabel("Vatnafræðilegt ár", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "07a_leitni_arleg.png", dpi=150)
plt.close()
print("\nMynd vistuð: figures/07a_leitni_arleg.png")


# ── Mynd 2: Mánaðarleg leitni – hitalíkamynd ─────────────────────────────────
#
# Litakvarði:
#   Blár   = marktæk lækkun  (p < 0.05, slope < 0)
#   Rauður = marktæk hækkun  (p < 0.05, slope > 0)
#   Grátt  = ekki marktækt

slopes_mo = np.array([mo_results[m]["slope"] for m in range(1, 13)])
pvals_mo  = np.array([mo_results[m]["p"]     for m in range(1, 13)])
sig_mo    = pvals_mo < 0.05

fig, ax = plt.subplots(figsize=(10, 4))

bar_colors = []
for s, p in zip(slopes_mo, pvals_mo):
    if p < 0.05 and s > 0:
        bar_colors.append("#d7191c")
    elif p < 0.05 and s < 0:
        bar_colors.append("#2c7bb6")
    else:
        bar_colors.append("#aaaaaa")

x = np.arange(1, 13)
bars = ax.bar(x, slopes_mo, color=bar_colors, edgecolor="white", linewidth=0.8)
ax.axhline(0, color="k", linewidth=0.8)

# p-gildi við hverja súlu – ofan á jákvæðar, neðan á neikvæðar
for i, (bar, p, s) in enumerate(zip(bars, pvals_mo, sig_mo)):
    h = bar.get_height()
    positive = h >= 0
    y_pos = h + 0.015 if positive else h - 0.015
    va    = "bottom"  if positive else "top"
    txt   = f"p={p:.3f}*" if s else f"p={p:.2f}"
    kw    = dict(ha="center", va=va, fontsize=7.5 if s else 7)
    if s:
        kw["fontweight"] = "bold"
    else:
        kw["color"] = "#555555"
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, txt, **kw)

ax.set_xticks(x)
ax.set_xticklabels(manud_nofn, fontsize=10)
ax.set_ylabel("Theil-Sen hallatala [m³/s á ári]", fontsize=10)
ax.set_title("Mánaðarleg leitni í rennsli Q – Ölfusá, Selfoss  (modified Mann-Kendall)")
ax.spines[["top", "right"]].set_visible(False)

legend_items = [
    mpatches.Patch(color="#d7191c", label="Marktæk aukning (p < 0.05)"),
    mpatches.Patch(color="#2c7bb6", label="Marktæk lækkun (p < 0.05)"),
    mpatches.Patch(color="#aaaaaa", label="Ekki marktækt"),
]
ax.legend(handles=legend_items, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "07b_leitni_manadarlegt.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/07b_leitni_manadarlegt.png")


# ── Mynd 3: Árstíðaleitni ────────────────────────────────────────────────────
season_colors = ["#2c7bb6", "#1a9641", "#d7191c", "#fdae61"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
axes = axes.flatten()
fig.suptitle("Árstíðaleitni í rennsli Q – Ölfusá, Selfoss", fontsize=12, fontweight="bold")

for ax, (name, r), clr in zip(axes, season_results.items(), season_colors):
    yrs = np.arange(len(r["y"])) + ar_Q.index[0]
    ax.plot(yrs, r["y"], "o-", color=clr, ms=4, lw=1.2, alpha=0.8)

    fit = r["intercept"] + r["slope"] * r["x"]
    ls  = "-" if r["sig"] else "--"
    lbl = (f"Theil-Sen: {r['slope']:+.3f}/ár\n"
           f"p = {r['p']:.3f}{'  (marktækt*)' if r['sig'] else ''}")
    ax.plot(yrs, fit, ls, color="k", lw=1.8, label=lbl)

    ax.set_title(name, fontsize=10)
    ax.set_ylabel("Q [m³/s]", fontsize=9)
    ax.set_xlabel("Vatnafræðilegt ár", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "07c_leitni_arstimar.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/07c_leitni_arstimar.png")
