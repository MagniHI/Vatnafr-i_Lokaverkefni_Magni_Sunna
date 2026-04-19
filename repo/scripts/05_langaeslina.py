"""
Skref 5 – Langæislína rennslis (Flow Duration Curve)
=====================================================
Langæislínan sýnir hve oft (hlutfall tíma) rennsli er yfir ákveðnu gildi.

  - X-ás: Yfirfallstíðni [%]  (0 % = alltaf, 100 % = aldrei)
  - Y-ás: Rennsli Q [m³/s]

Þrjú lykilgildi:
  Q5  – hárennsli  : rennsli sem er náð eða farið yfir 5 % af tíma
  Q50 – miðgildi   : rennsli sem er náð eða farið yfir 50 % af tíma
  Q95 – lágrennsli : rennsli sem er náð eða farið yfir 95 % af tíma

Lögun ferilsins:
  - Flatur ferill  → stöðugt rennsli (stöðuvötn, grunnvatn, jöklar jafna út)
  - Brattur ferill → sveiflukennt rennsli (lítið geymslulag)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Slóðir ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Gögn ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "olfus_flaedi_93_23.csv", parse_dates=["date"])
df = df.set_index("date").sort_index()
Q  = df["Q_m3s"].dropna().values


# ════════════════════════════════════════════════════════════════════════════
# REIKNA LANGÆISLÍNU
# ════════════════════════════════════════════════════════════════════════════
#
# Aðferð:
#   1. Raða öllum daglegum rennslisgildrum í lækkandi röð
#   2. Reikna yfirfallstíðni fyrir hvert gildi:
#
#        P(i) = i / (N + 1)  × 100  [%]
#
#      þar sem i = röðunarnúmer (1 = hæsta) og N = heildarfjöldi gilda.
#      Þetta er Weibull plotting position – algengasta aðferðin fyrir FDC.
#
#   3. Finna Q5, Q50, Q95 með brúun (interpolation)

Q_sorted = np.sort(Q)[::-1]          # lækkandi röð
N        = len(Q_sorted)
exceedance = np.arange(1, N + 1) / (N + 1) * 100   # Weibull, [%]

# Lykilgildi með brúun
Q5  = np.interp(5,  exceedance, Q_sorted)
Q50 = np.interp(50, exceedance, Q_sorted)
Q95 = np.interp(95, exceedance, Q_sorted)

print(f"\n══ Langæislína – lykilgildi ══")
print(f"  Q5   (hárennsli,  5%)  = {Q5:.1f}  m³/s")
print(f"  Q50  (miðgildi,  50%)  = {Q50:.1f}  m³/s")
print(f"  Q95  (lágrennsli, 95%) = {Q95:.1f}  m³/s")
print(f"\n  Q5/Q95 hlutfall        = {Q5/Q95:.2f}  (mælikvarði á sveiflu)")
print(f"  Fjöldi daga í greiningu: {N}")


# ════════════════════════════════════════════════════════════════════════════
# MYND 1 – Langæislína (log-skali á y-ás)
# ════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

ax.semilogy(exceedance, Q_sorted, color="#2c7bb6", linewidth=1.8,
            label="Dagrennsli 1993–2023")

# Merkja Q5, Q50, Q95
for pct, qval, clr, lbl in [
    (5,  Q5,  "#d7191c", f"Q5  = {Q5:.0f} m³/s"),
    (50, Q50, "#1a9641", f"Q50 = {Q50:.0f} m³/s"),
    (95, Q95, "#fdae61", f"Q95 = {Q95:.0f} m³/s"),
]:
    ax.axvline(pct, color=clr, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(qval, color=clr, linestyle=":",  linewidth=1.0, alpha=0.8)
    ax.scatter([pct], [qval], color=clr, s=60, zorder=5, label=lbl)

ax.set_xlabel("Yfirfallstíðni [%]", fontsize=11)
ax.set_ylabel("Rennsli Q [m³/s]  (log-skali)", fontsize=11)
ax.set_title("Langæislína rennslis – Ölfusá, Selfoss  (1993–2023)", fontsize=12)
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(which="major", linestyle="-",  linewidth=0.5, alpha=0.5)
ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "05a_langaeslina.png", dpi=150)
plt.close()
print("\nMynd vistuð: figures/05a_langaeslina.png")


# ════════════════════════════════════════════════════════════════════════════
# MYND 2 – FDC skipt á árstíma (vatnafræðilegt ár: okt–sep)
# ════════════════════════════════════════════════════════════════════════════
#
# Sýnir hvernig FDC breytist eftir árstímum –
# gagnlegt til að sjá hvaða árstími stýrir hárennsli og lágrennsli.

seasons = {
    "Vetur (Des–Feb)": [12, 1, 2],
    "Vor   (Mar–Maí)": [3,  4, 5],
    "Sumar (Jún–Ágú)": [6,  7, 8],
    "Haust (Sep–Nóv)": [9, 10, 11],
}
colors_s = ["#2c7bb6", "#1a9641", "#d7191c", "#fdae61"]

fig, ax = plt.subplots(figsize=(10, 5))

for (name, months), clr in zip(seasons.items(), colors_s):
    mask    = df.index.month.isin(months)
    q_s     = df.loc[mask, "Q_m3s"].dropna().values
    q_s_srt = np.sort(q_s)[::-1]
    exc_s   = np.arange(1, len(q_s_srt) + 1) / (len(q_s_srt) + 1) * 100
    ax.semilogy(exc_s, q_s_srt, color=clr, linewidth=1.6, label=name)

ax.set_xlabel("Yfirfallstíðni [%]", fontsize=11)
ax.set_ylabel("Rennsli Q [m³/s]  (log-skali)", fontsize=11)
ax.set_title("Langæislína eftir árstímum – Ölfusá, Selfoss", fontsize=12)
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(which="major", linestyle="-",  linewidth=0.5, alpha=0.5)
ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "05b_langaeslina_arstimar.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/05b_langaeslina_arstimar.png")
