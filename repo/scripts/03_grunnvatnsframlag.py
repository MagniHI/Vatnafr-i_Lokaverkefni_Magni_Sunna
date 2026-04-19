"""
Skref 3 – Mat á grunnvatnsframlagi
===================================
Þrjár aðferðir:
  A) Baseflow separation með Lyne-Hollick síu (Ladson et al., 2013)
  B) Baseflow Index (BFI)
  C) Recession analysis – metin recession constant (k)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path

# ── Slóðir ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Hlaða inn gögnum ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "olfus_flaedi_93_23.csv", parse_dates=["date"])
df = df.set_index("date").sort_index()
Q  = df["Q_m3s"].values          # NumPy fylki – auðveldara fyrir filterin


# ════════════════════════════════════════════════════════════════════════════
# A) LYNE-HOLLICK DIGITAL FILTER  (Ladson et al., 2013)
# ════════════════════════════════════════════════════════════════════════════
#
# Hugmyndin: heildarrennsli = grunnvatn + afrennsli (quickflow).
# Sían skilgreinir AFRENNSLI (f) og fær þannig grunnvatn sem leif.
#
# Jafna:  f[t] = α·f[t-1] + (1+α)/2 · (Q[t] − Q[t-1])
#   þar sem:
#     f[t]  = afrennsli á degi t
#     α     = síufæribreyta (0.925 er staðlað gildi, Ladson 2013)
#
#   Ef f[t] < 0  →  setjum f[t] = 0   (afrennsli getur ekki verið neikvætt)
#   Grunnvatn:  b[t] = Q[t] − f[t]
#   Ef b[t] > Q[t]  →  setjum b[t] = Q[t]
#
# Síunni er beitt 3 sinnum (fram, aftur, fram) til að fjarlægja
# fasaflökt (phase distortion).

def lyne_hollick(Q, alpha=0.925, passes=3):
    """
    Lyne-Hollick síu – skilar grunnvatni (baseflow).
    Q     : 1-D NumPy fylki með dagrennsli
    alpha : síufæribreyta (0–1); hærra gildi = sléttara grunnvatn
    passes: fjöldi síupassa (venjulega 3)
    """
    n = len(Q)
    signal = Q.copy().astype(float)

    for p in range(passes):
        f = np.zeros(n)          # afrennsli í þessum passa

        # Stefna: oddatala passar = fram, jafntala = aftur
        forward = (p % 2 == 0)
        idx = range(1, n) if forward else range(n - 2, -1, -1)
        i0  = 0            if forward else n - 1

        f[i0] = signal[i0] * (1 - alpha) / 2   # upphafsskilyrði

        for i in idx:
            prev = i - 1 if forward else i + 1
            f[i] = alpha * f[prev] + (1 + alpha) / 2 * (signal[i] - signal[prev])
            if f[i] < 0:
                f[i] = 0.0

        # Grunnvatn = heildarrennsli − afrennsli, klippt við [0, Q]
        baseflow = np.clip(signal - f, 0, signal)

        # Næsti passi notar grunnvatnið sem inntaksmerki
        signal = baseflow

    return baseflow


bf = lyne_hollick(Q, alpha=0.925, passes=3)
qf = Q - bf          # afrennsli (quickflow)

# Bæta við DataFrame
df["grunnvatn"]  = bf
df["hradrennsli"] = qf


# ════════════════════════════════════════════════════════════════════════════
# B) BASEFLOW INDEX (BFI)
# ════════════════════════════════════════════════════════════════════════════
#
# BFI = summa(grunnvatn) / summa(heildarrennsli)
#
# Túlkun: gildi nálægt 1 þýðir að mest allt rennsli kemur úr grunnvatni;
#         gildi nálægt 0 þýðir að afrennsli ræður ríkjum.

BFI_heild = bf.sum() / Q.sum()
print(f"\nBFI (allt tímabilið): {BFI_heild:.3f}")

# BFI á hvert ár – sýnir hvort grunnvatnsframlag er stöðugt milli ára
bfi_ar = (df["grunnvatn"].resample("YS-OCT").sum() /
          df["Q_m3s"].resample("YS-OCT").sum())
print("\nBFI eftir árum:")
print(bfi_ar.to_string())


# ════════════════════════════════════════════════════════════════════════════
# C) RECESSION ANALYSIS – metin recession constant (k)
# ════════════════════════════════════════════════════════════════════════════
#
# Þegar rennsli lækkar jafnt án úrkomu fylgir það venjulega veldisvísisfall:
#
#   Q(t) = Q₀ · e^(−k·t)
#
#   þar sem:
#     Q₀ = rennsli við upphaf samdráttar
#     k  = recession constant  [1/dagur]
#     t  = tími frá upphafi [dagar]
#
# Aðferð:
#   1. Finna "recession segments" – samfelldar raðir þar sem Q lækkar
#      og P = 0 (enginn úrkomu-áhrif).
#   2. Passa línu í log(Q) vs t  →  hallatala = −k.
#   3. Miðgildi k yfir öll brot gefur dæmigerða recession constant.

vedur = pd.read_csv(DATA_DIR / "olfus_vedur_93_23.csv", parse_dates=["date"])
vedur = vedur.set_index("date")
df = df.join(vedur[["P_mm"]], how="left")

MIN_LEN    = 5     # minnsta fjöldi daga í gildu recession broti
P_THRESH   = 0.5   # mm/dag – þykist enginn úrkoma ef P < þröskuldi

segments = []      # listi af (t-vigur, log(Q)-vigur) pörum

i = 0
n = len(df)
dagar = df.index

while i < n - MIN_LEN:
    # Hefst samdráttarhluti?
    if (df["Q_m3s"].iloc[i + 1] < df["Q_m3s"].iloc[i] and
            df["P_mm"].iloc[i] < P_THRESH):

        j = i + 1
        # Lengja brotið meðan rennsli lækkar og enginn úrkoma
        while (j < n - 1 and
               df["Q_m3s"].iloc[j + 1] < df["Q_m3s"].iloc[j] and
               df["P_mm"].iloc[j] < P_THRESH):
            j += 1

        seg_len = j - i
        if seg_len >= MIN_LEN:
            t_vec    = np.arange(seg_len + 1, dtype=float)
            q_seg    = df["Q_m3s"].iloc[i: j + 1].values
            logq_seg = np.log(q_seg)
            segments.append((t_vec, logq_seg))
        i = j + 1  # hoppa yfir þetta brot (dagur j er þegar meðhöndlaður)
    else:
        i += 1

print(f"\nFjöldi recession brota fundinn: {len(segments)}")

# Línuleg aðhvarfsgreining í log(Q) ~ t fyrir hvert brot
k_gildi = []
for t_vec, logq_seg in segments:
    slope, intercept, r, p, se = stats.linregress(t_vec, logq_seg)
    if slope < 0:                     # k verður að vera jákvætt
        k_gildi.append(-slope)

k_median  = np.median(k_gildi)
k_mean    = np.mean(k_gildi)
tau       = 1 / k_median              # geymslufastin (e-falt tímabil, e. storage coefficient)
half_life = np.log(2) / k_median      # halftími = ln(2)/k  [dagar]

print(f"\nRecession constant  k  (miðgildi): {k_median:.5f}  [1/dag]")
print(f"Recession constant  k  (meðaltal): {k_mean:.5f}  [1/dag]")
print(f"Geymslufastin       τ  = 1/k     : {tau:.1f}  dagar")
print(f"Halftími            t½ = ln2/k   : {half_life:.1f}  dagar")

# Vista k-gildi til notkunar í skriptu 08 (til að forðast harðkóðun)
recession_params = {"k_median": float(k_median), "k_mean": float(k_mean)}
with open(DATA_DIR / "recession_params.json", "w") as f:
    json.dump(recession_params, f, indent=2)
print(f"  → k-gildi vistuð í data/recession_params.json")


# ════════════════════════════════════════════════════════════════════════════
# MYNDIR
# ════════════════════════════════════════════════════════════════════════════

# ── Mynd 1: Baseflow separation – dæmiár (vatnafræðilegt ár 2010) ────────────
fig, ax = plt.subplots(figsize=(12, 4))

ar_syna = "2010"
mask = df.index.year == int(ar_syna)
dags = df.index[mask]

ax.fill_between(dags, df["grunnvatn"][mask], alpha=0.55,
                color="#2c7bb6", label="Grunnvatn (baseflow)")
ax.fill_between(dags, df["Q_m3s"][mask], df["grunnvatn"][mask],
                alpha=0.45, color="#d7191c", label="Afrennsli (quickflow)")
ax.plot(dags, df["Q_m3s"][mask], color="k", linewidth=0.8, label="Heildarrennsli Q")

ax.set_title(f"Baseflow separation – Lyne-Hollick síu  (α = 0.925) | {ar_syna}")
ax.set_ylabel("Rennsli [m³/s]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "03a_baseflow_separation.png", dpi=150)
plt.close()
print("\nMynd vistuð: figures/03a_baseflow_separation.png")

# ── Mynd 2: BFI eftir árum ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

ar_index = bfi_ar.index.year
ax.bar(ar_index, bfi_ar.values, color="#2c7bb6", alpha=0.75, edgecolor="white")
ax.axhline(BFI_heild, color="k", linestyle="--", linewidth=1.2,
           label=f"BFI heild = {BFI_heild:.3f}")
ax.set_title("Baseflow Index (BFI) eftir árum – Ölfusá, Selfoss")
ax.set_xlabel("Ár")
ax.set_ylabel("BFI")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "03b_bfi_eftir_arum.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/03b_bfi_eftir_arum.png")

# ── Mynd 3: Recession analysis – dreifing k-gilda og dæmisbrot ──────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Vinstri: dreifing k-gilda
axes[0].hist(k_gildi, bins=30, color="#2c7bb6", edgecolor="white", alpha=0.8)
axes[0].axvline(k_median, color="k", linestyle="--",
                label=f"Miðgildi k = {k_median:.4f}")
axes[0].set_title("Dreifing recession constant (k)")
axes[0].set_xlabel("k  [1/dag]")
axes[0].set_ylabel("Fjöldi brota")
axes[0].legend(fontsize=9)
axes[0].spines[["top", "right"]].set_visible(False)

# Hægri: eitt dæmisbrot á log-skala
best_seg = max(segments, key=lambda s: len(s[0]))
t_b, lq_b = best_seg
slope_b, intercept_b, *_ = stats.linregress(t_b, lq_b)
fit_line = intercept_b + slope_b * t_b

axes[1].scatter(t_b, lq_b, s=20, color="#2c7bb6", zorder=3, label="log(Q) mælt")
axes[1].plot(t_b, fit_line, color="#d7191c", linewidth=1.5,
             label=f"Aðhvarf  (k = {-slope_b:.4f})")
axes[1].set_title("Dæmisbrot – lengsta recession hluti")
axes[1].set_xlabel("Tími frá upphafi [dagar]")
axes[1].set_ylabel("log(Q)  [log m³/s]")
axes[1].legend(fontsize=9)
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "03c_recession_analysis.png", dpi=150)
plt.close()
print("Mynd vistuð: figures/03c_recession_analysis.png")
