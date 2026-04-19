"""
01_lysing_vatnasviðs.py
------------------------
Sækir eiginleika vatnasviðsins fyrir Ölfusá við Selfoss (ID 98) og framleiðir:

    figures/01a_landnytting.png  – súlurit af landnýtingarhlutföllum
    figures/01b_eiginleikar.png  – samantektartafla sem mynd

Prentar einnig samantekt í terminal til notkunar í skýrslu.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── slóðir ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
ATTR    = ROOT / "data" / "olfus_eiginleikar.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── hlaða eiginleikum ──────────────────────────────────────────────────────
attrs = pd.read_csv(ATTR, index_col="id")
a = attrs.iloc[0]

# ── terminal samantekt ─────────────────────────────────────────────────────
print("=" * 55)
print("  Ölfusá, Selfoss  –  LamaH-Ice ID 98")
print("=" * 55)
print(f"  Flatarmál                : {float(a['area_calc']):.1f} km²")
print(f"  Meðalhæð                 : {float(a['elev_mean']):.1f} m")
print(f"  Miðgildi hæðar           : {float(a['elev_med']):.1f} m")
print(f"  Hæðarbil                 : {float(a['elev_ran']):.1f} m")
print(f"  Meðalhalli               : {float(a['slope_mean']):.1f} m/km")
print(f"  Löngungarstuðull         : {float(a['elon_ratio']):.3f}")
print(f"  Lækjaþéttleiki           : {float(a['strm_dens']):.2f} km/km²")
print()
print("  Veðurfar og rennsli")
print(f"  Meðalúrkoma (CARRA)      : {float(a['p_mean']):.2f} mm/dag")
print(f"  Rennslisstuðull          : {float(a['runoff_ratio']):.3f}")
print(f"  Grunnvatnshlutfall (BFI) : {float(a['baseflow_index_ladson']):.3f}")
print(f"  Snjóhlutfall             : {float(a['frac_snow']):.2f}")
print()
print("  Landgerð")
print(f"  Jökull                   : {float(a['glac_fra'])*100:.1f} %")
print(f"  Bert land / hraunbreiður : {float(a['bare_fra'])*100:.1f} %")
print(f"  Kjarr og lyngheiðar      : {float(a['scrub_fra'])*100:.1f} %")
print(f"  Votlendi                 : {float(a['wetl_fra'])*100:.1f} %")
print(f"  Landbúnaður              : {float(a['agr_fra'])*100:.1f} %")
print(f"  Skógur                   : {float(a['forest_fra'])*100:.1f} %")
print(f"  Stöðuvatn                : {float(a['lake_fra'])*100:.1f} %")
print(f"  Þéttbýli                 : {float(a['urban_fra'])*100:.1f} %")
print()
print("  Jökull í smáatriðum")
print(f"  Flatarmál jökuls         : {float(a['g_area']):.1f} km²")
print(f"  Meðalhæð jökuls          : {float(a['g_mean_el']):.1f} m")
print(f"  Lágmarkshæð jökuls       : {float(a['g_min_el']):.1f} m")
print(f"  Hámarkshæð jökuls        : {float(a['g_max_el']):.1f} m")
print()
print("  Ríkjandi jarðfræði       : gosbasalt (vb)")
print(f"  Ríkjandi NI jarðfræði   : {a['g_dom_NI']}  (hraunbreiður)")
print(f"  Mannleg áhrif            : {a['degimpact']} / tegund {a['typimpact']}")
print("=" * 55)


# ── Mynd 1a: Súlurit landgerðar ─────────────────────────────────────────
landnytting = {
    "Bert land/hraun"  : float(a["bare_fra"])   * 100,
    "Kjarr/lyngheiðar" : float(a["scrub_fra"])  * 100,
    "Jökull"           : float(a["glac_fra"])   * 100,
    "Votlendi"         : float(a["wetl_fra"])   * 100,
    "Landbúnaður"      : float(a["agr_fra"])    * 100,
    "Stöðuvatn"        : float(a["lake_fra"])   * 100,
    "Skógur"           : float(a["forest_fra"]) * 100,
    "Þéttbýli"         : float(a["urban_fra"])  * 100,
}

colors = ["#b5b5b5", "#8fbc8f", "#add8e6", "#6b9aad", "#f5deb3", "#4682b4", "#228b22", "#c0392b"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(landnytting.keys(), landnytting.values(), color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, landnytting.values()):
    if val > 0.5:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Hlutfall vatnasviðs (%)", fontsize=11)
ax.set_title("Ölfusá við Selfoss (ID 98) – Landgerð", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(landnytting.values()) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "01a_landnytting.png", dpi=150, bbox_inches="tight")
plt.close()
print("Vistað → figures/01a_landnytting.png")


# ── Mynd 1b: Samantektartafla ──────────────────────────────────────────────
tafla_gögn = [
    ["Flatarmál",              f"{float(a['area_calc']):.1f} km²"],
    ["Meðalhæð",               f"{float(a['elev_mean']):.1f} m"],
    ["Miðgildi hæðar",         f"{float(a['elev_med']):.1f} m"],
    ["Hæðarbil",               f"{float(a['elev_ran']):.1f} m"],
    ["Meðalhalli",             f"{float(a['slope_mean']):.1f} m/km"],
    ["Löngungarstuðull",       f"{float(a['elon_ratio']):.3f}"],
    ["Meðalúrkoma",            f"{float(a['p_mean']):.2f} mm/dag"],
    ["Rennslisstuðull",        f"{float(a['runoff_ratio']):.3f}"],
    ["Grunnvatnshlutfall (BFI)", f"{float(a['baseflow_index_ladson']):.3f}"],
    ["Snjóhlutfall",           f"{float(a['frac_snow']):.2f}"],
    ["Jökulshlutfall",         f"{float(a['glac_fra'])*100:.1f} %"],
    ["Flatarmál jökuls",       f"{float(a['g_area']):.1f} km²"],
    ["Ríkjandi jarðfræði",     "Gosbasalt (vb)"],
    ["Mannleg áhrif",          f"Lítil ({a['degimpact']}) / tegund {a['typimpact']}"],
]

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.axis("off")

tbl = ax.table(
    cellText=tafla_gögn,
    colLabels=["Eiginleiki", "Gildi"],
    loc="center",
    cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5)

# Stíll á hauslínu
for col in range(2):
    tbl[(0, col)].set_facecolor("#2c5f8a")
    tbl[(0, col)].set_text_props(color="white", fontweight="bold")

# Skiptilegar litaskipanir á línum
for row in range(1, len(tafla_gögn) + 1):
    bg = "#f0f4f8" if row % 2 == 0 else "white"
    for col in range(2):
        tbl[(row, col)].set_facecolor(bg)

ax.set_title("Ölfusá við Selfoss (ID 98)\nEiginleikar vatnasviðs",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "01b_eiginleikar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Vistað → figures/01b_eiginleikar.png")

print("\nLokið.")
