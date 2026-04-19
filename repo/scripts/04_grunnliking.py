"""
Skref 4 – Tenging við grunnlíkingu vatnafræðinnar
===================================================
Grunnlíkingin:

    P = Q + ET + ΔS

  þar sem:
    P   = úrkoma  [mm/dag]           ← reiknuð úr CARRA endurgreiningu
    Q   = afrennsli [mm/dag]         ← mælt rennsli umreiknað yfir í dýptareiningar
    ET  = uppgufun [mm/dag]         ← reiknuð úr CARRA (total_et_carra)
    ΔS  = breyting í geymslu [mm/dag]← leifð (óþekkt, reiknuð sem P − Q − ET)

Jákvætt ΔS þýðir aukning í geymslu (snjór, jökull, grunnvatn).
Neikvætt ΔS þýðir að geymsla minnkar – t.d. bræðing jökuls bætist við afrennsli.
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

AREA_KM2 = 5724.193   # flatarmál vatnasviðs [km²]  – úr Catchment_attributes.csv

# Stuðull til að breyta Q [m³/s] yfir í [mm/dag]
# Q [m³/s] × 86400 [s/dag] / (A [km²] × 1e6 [m²/km²]) × 1000 [mm/m]
Q_TO_MM = 86400 / (AREA_KM2 * 1e6) * 1000


# ════════════════════════════════════════════════════════════════════════════
# HLAÐA INN GÖGNUM
# ════════════════════════════════════════════════════════════════════════════

# Mælt rennsli
q_df = pd.read_csv(DATA_DIR / "olfus_flaedi_93_23.csv", parse_dates=["date"])
q_df = q_df.set_index("date")

# Veðurgögn (P og T úr hreinsaðri skrá)
v_df = pd.read_csv(DATA_DIR / "olfus_vedur_93_23.csv", parse_dates=["date"])
v_df = v_df.set_index("date")

# Sækja ET_carra og PET úr upprunalegu gögnum
# (total_et_carra er í mm/dag, reiknað yfir allt vatnasviðið í CARRA)
raw = pd.read_csv(
    DATA_DIR / "ID_98_OLF_.VEDURcsv.csv",
    sep=";",
    parse_dates={"date": ["YYYY", "MM", "DD"]},
    date_format="%Y %m %d",
)
raw = raw.set_index("date")
raw = raw.loc["1993-10-01":"2023-09-30"]

# Sameina allt í eitt DataFrame
df = pd.DataFrame(index=q_df.index)
df["Q_m3s"]  = q_df["Q_m3s"]
df["P_mm"]   = v_df["P_mm"]                        # CARRA úrkoma [mm/dag]
df["T_degC"] = v_df["T_degC"]                       # CARRA hitastig [°C]
df["ET_mm"]  = raw["total_et_carra"].reindex(df.index)   # CARRA ET [mm/dag]
df["PET_mm"] = raw["pet"].reindex(df.index)         # ERA5-L PET [mm/dag] – til samanburðar

# ── Umreikningur Q → mm/dag ──────────────────────────────────────────────────
#
# Rennsli er mælt í m³/s en þarf að vera í mm/dag til að bera saman við P og ET.
# Við deilum með flatarmáli vatnasviðsins og margföldum með dýptarstuðli:
#
#   Q [mm/dag] = Q [m³/s] × 86400 [s/dag] / A [m²] × 1000 [mm/m]
#
df["Q_mm"] = df["Q_m3s"] * Q_TO_MM


# ════════════════════════════════════════════════════════════════════════════
# GRUNNLÍKINGIN
# ════════════════════════════════════════════════════════════════════════════
#
# P = Q + ET + ΔS
#
# Við þekkjum P, Q og ET → reiknum ΔS sem leifð:
#
#   ΔS = P − Q − ET
#
# Á langtímabili (30 ár) á ΔS að vera nálægt núlli ef geymsla er stöðug.
# Neikvætt meðal-ΔS bendir til þess að jökullinn sé að minnka (bræðing).

df["dS_mm"] = df["P_mm"] - df["Q_mm"] - df["ET_mm"]


# ── Árssamantekt ─────────────────────────────────────────────────────────────
# Notum vatnafræðilegt ár (okt–sep) til að halda samræmi
ar = pd.DataFrame({
    "P_mm" : df["P_mm"].resample("YS-OCT").sum(),
    "Q_mm" : df["Q_mm"].resample("YS-OCT").sum(),
    "ET_mm": df["ET_mm"].resample("YS-OCT").sum(),
    "dS_mm": df["dS_mm"].resample("YS-OCT").sum(),
})
ar = ar.dropna()
ar.index = ar.index.year     # nota upphafsar sem stika

# Meðaltöl yfir allt tímabilið
mean_P  = ar["P_mm"].mean()
mean_Q  = ar["Q_mm"].mean()
mean_ET = ar["ET_mm"].mean()
mean_dS = ar["dS_mm"].mean()

print("\n══ Meðal-ársjafnagjald [mm/ár] ══")
print(f"  P   (úrkoma,   CARRA)    = {mean_P:.1f}")
print(f"  Q   (afrennsli, mælt)    = {mean_Q:.1f}")
print(f"  ET  (uppgufun, CARRA)   = {mean_ET:.1f}")
print(f"  ΔS  (geymslubreyting)    = {mean_dS:.1f}   ← P − Q − ET")
print(f"\n  Rennslisstuðull Q/P      = {mean_Q/mean_P:.3f}")

# ── Mánaðarmeðaltöl ───────────────────────────────────────────────────────────
mo = pd.DataFrame({
    "P_mm" : df["P_mm"].groupby(df.index.month).mean(),
    "Q_mm" : df["Q_mm"].groupby(df.index.month).mean(),
    "ET_mm": df["ET_mm"].groupby(df.index.month).mean(),
    "dS_mm": df["dS_mm"].groupby(df.index.month).mean(),
})

print("\n══ Mánaðarmeðaltöl [mm/dag] ══")
print(mo.round(2).to_string())


# ════════════════════════════════════════════════════════════════════════════
# ÓVISSA Í HVERJUM LIÐ
# ════════════════════════════════════════════════════════════════════════════
print("""
══ Óvissumatsþáttkar ══
  P  (CARRA):  Reiknað úr endurgreiningu – óvissa vegna staðfæringar og
               hæðarleiðréttingar; ekki bein mæling á vatnasviðinu.
               Metin óvissa: ±10–20%

  Q  (mælt):   Mælt rennsli við Selfoss – bein mæling en með óvissu vegna
               ratingfalls og gæðamerkja; talinn áreiðanlegasti liðurinn.
               Metin óvissa: ±5%

  ET (CARRA):  Reiknað úr orkujafnaðri CARRA – mest óviss stærðin.
               Engar beinnar ET-mælingar á vatnasviðinu.
               Metin óvissa: ±20–40%

  ΔS (leifð):  Óþekkt – inniheldur geymslubreytingar í jökli, snjó og
               grunnvatni. Ekki mælt beint. Gæti endurspeglað jökulbræðslu.
""")


# ════════════════════════════════════════════════════════════════════════════
# MYNDIR
# ════════════════════════════════════════════════════════════════════════════

manud_nofn = ["Okt","Nóv","Des","Jan","Feb","Mar","Apr","Maí","Jún","Júl","Ágú","Sep"]
x = np.arange(1, 13)

# Raðum í vatnafræðilegt ár (okt = 1)
def hydro_order(series):
    order = [10,11,12,1,2,3,4,5,6,7,8,9]
    return series.reindex(order).values

P_mo  = hydro_order(mo["P_mm"])
Q_mo  = hydro_order(mo["Q_mm"])
ET_mo = hydro_order(mo["ET_mm"])
dS_mo = hydro_order(mo["dS_mm"])

# ── Mynd 1: Mánaðarmeðaltöl – staflarit ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

# Q er neðst, ET ofan á Q, ΔS (jákvætt) ofan á ET
# Neikvætt ΔS sýnir bræðsluframlag (lækkar nettó geymslu)
ax.bar(x, Q_mo,  color="#2c7bb6", alpha=0.85, label="Q – afrennsli (mælt)")
ax.bar(x, ET_mo, color="#fdae61", alpha=0.85, bottom=Q_mo,
       label="ET – uppgufun (CARRA)")

# ΔS jákvætt = geymsla hleðst (t.d. snjófall); neikvætt = geymsla tæmist
pos_dS = np.where(dS_mo > 0, dS_mo, 0)
neg_dS = np.where(dS_mo < 0, dS_mo, 0)
ax.bar(x, pos_dS, color="#1a9641", alpha=0.75, bottom=Q_mo + ET_mo,
       label="ΔS+ – geysluhleðsla")
ax.bar(x, neg_dS, color="#d7191c", alpha=0.75, bottom=Q_mo + ET_mo,
       label="ΔS− – geymslutæming")

# P sem línu ofan á stöplunum
ax.plot(x, P_mo, "k-o", linewidth=2, markersize=5, zorder=5,
        label="P – úrkoma (CARRA)")

ax.set_xticks(x)
ax.set_xticklabels(manud_nofn, fontsize=10)
ax.set_ylabel("mm/dag")
ax.set_title("Vatnsbúskapur – Ölfusá, Selfoss  (1993–2023)\n"
             "P = Q + ET + ΔS")
ax.legend(fontsize=9, loc="upper right")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "04a_grunnliking_manadarlegt.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/04a_grunnliking_manadarlegt.png")


# ── Mynd 2: Árlegar stærðir með lóðréttum súlum ─────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ar_idx = ar.index

# Efri mynd: P, Q, ET
axes[0].plot(ar_idx, ar["P_mm"],  "b-",  lw=1.5, label="P – úrkoma")
axes[0].plot(ar_idx, ar["Q_mm"],  "k-",  lw=1.5, label="Q – afrennsli")
axes[0].plot(ar_idx, ar["ET_mm"], "r-",  lw=1.5, label="ET – uppgufun")
axes[0].scatter(ar_idx, ar["P_mm"],  color="b", s=28, alpha=0.35, zorder=3)
axes[0].scatter(ar_idx, ar["Q_mm"],  color="k", s=28, alpha=0.35, zorder=3)
axes[0].scatter(ar_idx, ar["ET_mm"], color="r", s=28, alpha=0.35, zorder=3)
axes[0].set_ylabel("mm/ár")
axes[0].set_title("Ársleg vatnsbúskapur – Ölfusá, Selfoss")
axes[0].legend(fontsize=9)
axes[0].spines[["top","right"]].set_visible(False)
axes[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())

# Neðri mynd: ΔS (geymslubeyting)
colors = ["#d7191c" if v < 0 else "#1a9641" for v in ar["dS_mm"]]
axes[1].bar(ar_idx, ar["dS_mm"], color=colors, alpha=0.8, width=0.7)
axes[1].axhline(0, color="k", linewidth=0.8)
axes[1].axhline(mean_dS, color="gray", linestyle="--", lw=1.2,
                label=f"Meðal ΔS = {mean_dS:.0f} mm/ár")
axes[1].set_ylabel("ΔS  [mm/ár]")
axes[1].set_xlabel("Vatnafræðilegt ár")
axes[1].set_title("Geymslubreyting ΔS = P − Q − ET")
axes[1].legend(fontsize=9)
axes[1].spines[["top","right"]].set_visible(False)

# X-ás: stór stika á 5 ára millibili, lítil stika á hverju ári
axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axes[1].tick_params(axis="x", which="minor", length=4, color="gray")

plt.tight_layout()
plt.savefig(FIG_DIR / "04b_grunnliking_arleg.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/04b_grunnliking_arleg.png")
