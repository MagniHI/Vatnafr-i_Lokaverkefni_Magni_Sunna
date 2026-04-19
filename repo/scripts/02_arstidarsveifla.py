"""
02_arstidarsveifla.py
----------------------
Reiknar meðaltalsár (climatology) fyrir rennsli, úrkomu og hitastig
yfir tímabilið 1993-10-01 – 2023-09-30 og framleiðir:

    figures/02_arstidarsveifla.png  – þríhluta mynd (Q, P, T)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── slóðir ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── hlaða gögnum ───────────────────────────────────────────────────────────
flow = pd.read_csv(ROOT / "data" / "olfus_flaedi_93_23.csv",
                   index_col="date", parse_dates=True)
met  = pd.read_csv(ROOT / "data" / "olfus_vedur_93_23.csv",
                   index_col="date", parse_dates=True)

df = flow.join(met)

# ── meðaltalsár (climatology) ──────────────────────────────────────────────
# Flokka eftir mánuði og reikna meðaltal yfir öll 30 ár
månuðir = range(1, 13)
manud_nofn = ["Jan", "Feb", "Mar", "Apr", "Maí", "Jún",
              "Júl", "Ágú", "Sep", "Okt", "Nóv", "Des"]

clim     = df.groupby(df.index.month).mean()
clim_std = df.groupby(df.index.month).std()

Q_mean = clim["Q_m3s"]
Q_std  = clim_std["Q_m3s"]

# Mánaðarsummur úrkomu (ein gildi á mánuð × ár) – rétt leið til að fá std mánaðarsummu
monthly_P_totals = df["P_mm"].resample("ME").sum()
P_mean = monthly_P_totals.groupby(monthly_P_totals.index.month).mean()
P_std  = monthly_P_totals.groupby(monthly_P_totals.index.month).std()

T_mean = clim["T_degC"]
T_std  = clim_std["T_degC"]

x = list(månuðir)

# ── mynd ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("Ölfusá við Selfoss (ID 98)\nMeðaltalsár 1993–2023",
             fontsize=13, fontweight="bold", y=0.98)

# --- Rennsli (Q) ---
ax1 = axes[0]
ax1.plot(x, Q_mean, color="#1a6faf", linewidth=2.2, marker="o", markersize=4, zorder=3)
ax1.fill_between(x,
                 Q_mean - Q_std,
                 Q_mean + Q_std,
                 color="#1a6faf", alpha=0.15, label="±1 staðalfrávik")
ax1.set_ylabel("Rennsli (m³/s)", fontsize=10)
ax1.legend(fontsize=9, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# --- Úrkoma (P) ---
ax2 = axes[1]
bar_colors = ["#2196F3" if p >= P_mean.mean() else "#90CAF9" for p in P_mean]
bars = ax2.bar(x, P_mean, color=bar_colors, edgecolor="white", linewidth=0.6, zorder=3)
ax2.errorbar(x, P_mean, yerr=P_std, fmt="none", color="black",
             capsize=3, linewidth=0.8, zorder=4)
ax2.set_ylabel("Úrkoma (mm/mán)", fontsize=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

# --- Hitastig (T) ---
ax3 = axes[2]
ax3.bar(x, T_mean.clip(lower=0), color="#e74c3c", edgecolor="white",
        linewidth=0.6, zorder=3, label="Yfir frostmarki")
ax3.bar(x, T_mean.clip(upper=0), color="#3498db", edgecolor="white",
        linewidth=0.6, zorder=3, label="Undir frostmarki")
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.fill_between(x,
                 T_mean - T_std,
                 T_mean + T_std,
                 color="gray", alpha=0.15)
ax3.set_ylabel("Hitastig (°C)", fontsize=10)
ax3.legend(fontsize=9, loc="upper left")
ax3.spines[["top", "right"]].set_visible(False)

# Lína fyrir hverja gráðu, þykkari á 5 gráðu millibili
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax3.grid(axis="y", which="major", linestyle="-",  linewidth=0.9, alpha=0.7)
ax3.grid(axis="y", which="minor", linestyle="--",  linewidth=0.5, alpha=0.6)

# --- x-ás merkingar ---
ax3.set_xticks(x)
ax3.set_xticklabels(manud_nofn, fontsize=10)
ax3.set_xlim(0.5, 12.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "02_arstidarsveifla.png", dpi=150, bbox_inches="tight")
plt.close()
print("Vistað → figures/02_arstidarsveifla.png")

# ── prenta tölu samantekt ──────────────────────────────────────────────────
print("\nMeðaltalsár – mánaðarleg gildi:")
print(f"{'Mánuður':<8} {'Q (m³/s)':>10} {'P (mm/mán)':>12} {'T (°C)':>8}")
print("-" * 42)
for i, m in enumerate(manud_nofn):
    print(f"{m:<8} {Q_mean.iloc[i]:>10.1f} {P_mean.iloc[i]:>12.1f} {T_mean.iloc[i]:>8.1f}")

print(f"\nÁrsmeðaltal Q : {Q_mean.mean():.1f} m³/s")
print(f"Árssumma P    : {P_mean.sum():.0f} mm/ár")
print(f"Árleg meðalT  : {T_mean.mean():.1f} °C")
print(f"(P_mean er í mm/mán, P_std er staðalfrávik mánaðarsumma yfir 30 ár)")
print("\nLokið.")
