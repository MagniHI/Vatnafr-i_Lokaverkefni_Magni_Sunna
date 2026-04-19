"""
Skref 8 – Greining á rennslisatburði
======================================
Valinn atburður: stærsta flóð á mælingatímabilinu
  Qmax = 1930.9 m³/s  –  21. desember 2006
  Tegund: hlýviðurflóð / rain-on-snow

Greiningartímabil: 6. des 2006 – 25. jan 2007  (extended til að sýna samdrátt)

Mælt á myndinni:
  • Time-to-peak         – tími frá upphafi rennslisaukningar að Qmax
  • Excess rain time     – tími frá lokum meginúrkomu þangað til Q_base  (metið)
  • Recession time       – tími frá Qpeak þangað til Q_base  (metið með k)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# GÖGN – víðara gluggi til að sýna samdrátt
# ════════════════════════════════════════════════════════════════════════════
flow = pd.read_csv(DATA_DIR / "olfus_flaedi_93_23.csv",
                   parse_dates=["date"]).set_index("date")
met  = pd.read_csv(DATA_DIR / "olfus_vedur_93_23.csv",
                   parse_dates=["date"]).set_index("date")
df   = flow.join(met)

WIN_START = pd.Timestamp("2006-12-06")
WIN_END   = pd.Timestamp("2007-01-25")
ev = df.loc[WIN_START:WIN_END].copy()

Q = ev["Q_m3s"]
P = ev["P_mm"]
T = ev["T_degC"]

# ════════════════════════════════════════════════════════════════════════════
# LYKILSTIGI
# ════════════════════════════════════════════════════════════════════════════
Q_base   = Q.loc["2006-12-14":"2006-12-17"].mean()   # ~267 m³/s
t_rise   = pd.Timestamp("2006-12-18")                 # upphaf aukningar
t_peak   = Q.idxmax()                                  # 21. des
Q_peak   = Q.max()
t_P_end  = pd.Timestamp("2006-12-22")                 # lokur meginbylgju
ttp_days = (t_peak - t_rise).days                      # time-to-peak

with open(DATA_DIR / "recession_params.json") as f:
    k_rec = json.load(f)["k_median"]                   # [1/dag], lesið úr skref 3
t_recession_days  = np.log(Q_peak / Q_base) / k_rec   # ~68 dagar
t_return          = t_peak + pd.Timedelta(days=t_recession_days)
excess_days       = (t_return - t_P_end).days

print("══ Lykilstærðir atburðarins ══")
print(f"  Grunnrennsli (Q_base)       : {Q_base:.1f} m³/s")
print(f"  Upphaf rennslisaukningar    : {t_rise.date()}")
print(f"  Topprennsli (Q_peak)        : {Q_peak:.1f} m³/s  ({t_peak.date()})")
print(f"  Time-to-peak                : {ttp_days} dagar")
print(f"  Lokur meginúrkomu           : {t_P_end.date()}")
print(f"  Recession time (metið)      : {t_recession_days:.1f} dagar → {t_return.date()}")
print(f"  Excess rain release (metið) : {excess_days} dagar")

# ════════════════════════════════════════════════════════════════════════════
# MYND
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [3, 1.5, 1], "hspace": 0.06})
ax1, ax2, ax3 = axes

fig.suptitle(
    "Rennslisatburður – Ölfusá, Selfoss  |  Desember 2006 – Janúar 2007\n"
    "Stærsta flóð á mælitímabilinu  (Q$_{max}$ = 1931 m³/s, 21. des 2006)",
    fontsize=12, fontweight="bold", y=1.00)

dags = ev.index

# ─── AX1: Rennsli ────────────────────────────────────────────────────────────
ax1.plot(dags, Q, color="#2c7bb6", linewidth=2.2, zorder=3,
         label="Rennsli Q [m³/s]")
ax1.fill_between(dags, Q, alpha=0.12, color="#2c7bb6")

# Grunnrennsli
ax1.axhline(Q_base, color="#1a9641", linestyle="--", linewidth=1.4,
            label=f"Grunnrennsli Q$_{{base}}$ = {Q_base:.0f} m³/s")

# Recession ferill – aðeins innan gluggans
t_rec_max = (WIN_END - t_peak).days + 1
t_rec_vec = np.linspace(0, t_rec_max, 300)
Q_rec     = Q_peak * np.exp(-k_rec * t_rec_vec)
dates_rec = [t_peak + pd.Timedelta(days=float(d)) for d in t_rec_vec]
ax1.plot(dates_rec, Q_rec, color="k", linestyle="--", linewidth=1.4, alpha=0.55,
         label=f"Metinn recession  Q = Q$_{{max}}$·e$^{{-kt}}$  (k = {k_rec:.4f} dag$^{{-1}}$)")

# Lóðréttar merkingarlínur (allar þrjár á öllum þremur ásum)
vkw = dict(linewidth=1.5, alpha=0.85, zorder=4)
for ax in [ax1, ax2, ax3]:
    ax.axvline(t_rise,  color="#e67e22", linestyle="-",   **vkw)
    ax.axvline(t_peak,  color="#d7191c", linestyle="-",   **vkw)
    ax.axvline(t_P_end, color="#8e44ad", linestyle="-.",  **vkw)

# ── Time-to-peak tvíörv (notum annotate með connectionstyle) ─────────────────
# Setjum hana í miðju við ~60% af Qmax til að gefa henni pláss
y_ttp = Q_peak * 0.62
ax1.annotate("", xy=(t_peak, y_ttp), xytext=(t_rise, y_ttp),
             arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=2.0))
mid_ttp = t_rise + (t_peak - t_rise) / 2
ax1.text(mid_ttp, y_ttp + Q_peak * 0.03,
         f"Time-to-peak\n{ttp_days} dagar",
         ha="center", va="bottom", fontsize=9, color="#e67e22", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e67e22", alpha=0.85))

# ── Recession time tvíörv (frá toppi að lok gluggans, með metið endagildi) ───
y_rec = Q_peak * 0.30
# Bogi frá toppi að WIN_END (t_return er utan gluggans)
ax1.annotate("", xy=(WIN_END, y_rec), xytext=(t_peak, y_rec),
             arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8))
mid_rec = t_peak + (WIN_END - t_peak) * 0.45
ax1.text(mid_rec, y_rec + Q_peak * 0.03,
         f"Recession time ≈ {t_recession_days:.0f} dagar\n→ Q$_{{base}}$ ~{t_return.strftime('%d. %b %Y')} (metið)",
         ha="center", va="bottom", fontsize=8.5, color="#555555",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", alpha=0.85))

# ── Excess rain release textbox (t_P_end → t_return, báðar utan/á brún) ─────
y_exc = Q_peak * 0.12
ax1.annotate("", xy=(WIN_END, y_exc), xytext=(t_P_end, y_exc),
             arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=1.8))
mid_exc = t_P_end + (WIN_END - t_P_end) * 0.4
ax1.text(mid_exc, y_exc + Q_peak * 0.03,
         f"Excess rain release ≈ {excess_days} dagar (metið)",
         ha="center", va="bottom", fontsize=8.5, color="#8e44ad",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8e44ad", alpha=0.85))

# ── Merkingartextar við lóðréttar línur (efst, utan flóðs) ───────────────────
txt_y = Q_peak * 1.02
ax1.text(t_rise  - pd.Timedelta(hours=10), txt_y,
         "Upphaf\naukningar", ha="right", va="bottom",
         fontsize=8, color="#e67e22")
ax1.text(t_peak  + pd.Timedelta(hours=6),  txt_y,
         f"Q$_{{max}}$ = {Q_peak:.0f} m³/s", ha="left", va="bottom",
         fontsize=9, color="#d7191c", fontweight="bold")
ax1.text(t_P_end + pd.Timedelta(hours=6),  Q_peak * 0.82,
         "Lokur\nmeginúrkomu", ha="left", va="top",
         fontsize=8, color="#8e44ad")

# 2. flóð merkið
ax1.annotate("2. flóð (62 mm, 24. des)",
             xy=(pd.Timestamp("2006-12-25"), Q.loc["2006-12-25"]),
             xytext=(pd.Timestamp("2007-01-02"), 1050),
             fontsize=8, color="#2980b9",
             arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.2),
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2980b9", alpha=0.8))

ax1.set_ylabel("Rennsli Q [m³/s]", fontsize=10)
ax1.set_ylim(bottom=0)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(axis="y", which="major", linestyle="-",  linewidth=0.4, alpha=0.4)
ax1.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.25)
ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(labelbottom=False)

# ─── AX2: Úrkoma ──────────────────────────────────────────────────────────────
# Litum stöplana eftir tegund: meginbylgja, 2. bylgja, aðrar
bar_colors = []
for d, p_val in zip(dags, P):
    if pd.Timestamp("2006-12-18") <= d <= t_P_end:
        bar_colors.append("#d7191c")      # meginbylgja
    elif pd.Timestamp("2006-12-24") <= d <= pd.Timestamp("2006-12-25"):
        bar_colors.append("#2980b9")      # 2. bylgja
    else:
        bar_colors.append("#74add1")      # önnur úrkoma

ax2.bar(dags, P, width=0.85, color=bar_colors, alpha=0.85)

# Bæta við daglegum gildum á stærstu stöplunum
for d, p_val in zip(dags, P):
    if p_val > 20:
        ax2.text(d, p_val + 1, f"{p_val:.0f}", ha="center", va="bottom",
                 fontsize=7.5, color="#333333")

# Legend handvirkt
legend_p = [
    mpatches.Patch(color="#d7191c", alpha=0.85, label="Meginbylgja (18.–22. des)"),
    mpatches.Patch(color="#2980b9", alpha=0.85, label="2. bylgja (24.–25. des)"),
    mpatches.Patch(color="#74add1", alpha=0.85, label="Önnur úrkoma"),
]
ax2.legend(handles=legend_p, fontsize=8, loc="upper right", framealpha=0.9)
ax2.set_ylabel("P [mm/dag]", fontsize=10)
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.grid(axis="y", which="major", linestyle="-", linewidth=0.4, alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelbottom=False)

# ─── AX3: Hitastig ────────────────────────────────────────────────────────────
ax3.plot(dags, T, color="#c0392b", linewidth=1.8, zorder=3)
ax3.fill_between(dags, T, 0, where=(T >= 0), color="#e74c3c", alpha=0.30,
                 label="T ≥ 0°C  (bræðsluskilyrði)")
ax3.fill_between(dags, T, 0, where=(T < 0),  color="#3498db", alpha=0.25,
                 label="T < 0°C  (frostmark)")
ax3.axhline(0, color="k", linewidth=1.0, linestyle="--", alpha=0.5)
ax3.set_ylabel("T [°C]", fontsize=10)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax3.grid(axis="y", which="major", linestyle="-",  linewidth=0.4, alpha=0.4)
ax3.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.25)
ax3.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
ax3.spines[["top", "right"]].set_visible(False)

# X-ás á neðsta plotti
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

# Lítil hök á x-ás á öllum þremur ásum
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=3, width=0.8, color="#555555")

# Samræma x-ás mörk nákvæmlega
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(WIN_START, WIN_END)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(FIG_DIR / "08_rennslisatburdur.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nMynd vistuð: figures/08_rennslisatburdur.png")

print("""
══ Samantekt til skýrslu ══

Tegund flóðs:
  Hlýviðurflóð (rain-on-snow). Í aðdragandanum var kalt (-7 til -9°C)
  og mikill snjór lagðist á vatnasviðið (6.–17. des). Þegar hlýtt
  rigndi 18.–21. des (T > 0°C, P: 23→53→80→19 mm/dag) bræddi bæði
  snjór og rigning saman og olli skyndilegri flóðbylgju.

Lögun vatnsritsins:
  • Brattur uppgangur (3 dagar frá 267→1931 m³/s) – lítil geymsluseinkun
  • Tiltölulega hraður samdráttarhluti þar til annað atvik kemur 24. des
  • Hátt BFI (0.80) kemur EKKI fram hér – flóðið er quickflow-dominerað

Tenging við BFI og FDC:
  • BFI=0.80 á við meðalskilyrði – í flóðum skiptast hlutföll;
    afrennsli ræður ríkjum þegar snjóbræðsla + rigning kemur saman
  • Q-peak (1931) er mun hærra en Q5 á FDC (582 m³/s) –
    þetta er mjög sjaldgæfur atburður (T ≈ 80–100 ár samkvæmt skref 6)
  • Recession time ~68 dagar er í samræmi við k=0.029 (τ=34.6 dagar)
    – langt samdráttartímabil vegna djúpra jarðvatnsgeymsla
""")
