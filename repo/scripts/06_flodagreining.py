"""
Skref 6 – Flóðagreining
========================
A) Flóðatíðni eftir mánuðum (flood seasonality)
B) Líkindadreifingargreining – árstoppar (annual peak flows):
     • Gumbel
     • Log Normal 3 stika (LN3)
     • Log Pearson 3 stika (LP3)
   Gringorten plotting positions, val á bestu dreifingu (lægst RMSE).
   Reiknar Q10, Q50, Q100 með 90% öryggisbili (bootstrap).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.optimize import minimize_scalar
from pathlib import Path

DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)


# ════════════════════════════════════════════════════════════════════════════
# GÖGN – ÁRSTOPPAR
# ════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_DIR / "olfus_flaedi_93_23.csv", parse_dates=["date"])
df = df.set_index("date").sort_index()

# Vatnafræðilegt ár: okt(t) – sep(t+1) → merkt sem ár t+1
df["vy"] = np.where(df.index.month >= 10, df.index.year + 1, df.index.year)

peaks_s  = df.groupby("vy")["Q_m3s"].agg(["max", "idxmax"])
peaks_s.columns = ["Q_peak", "date_peak"]
peaks_s  = peaks_s.dropna()

Q_peaks  = peaks_s["Q_peak"].values        # [m³/s]
months   = peaks_s["date_peak"].dt.month.values
N        = len(Q_peaks)
print(f"Fjöldi árstoppa: {N}")
print(f"Hæsti toppur: {Q_peaks.max():.1f} m³/s  ({peaks_s['date_peak'][peaks_s['Q_peak'].idxmax()].date()})")


# ════════════════════════════════════════════════════════════════════════════
# A) FLÓÐATÍÐNI EFTIR MÁNUÐUM
# ════════════════════════════════════════════════════════════════════════════
manud_nofn = ["Jan","Feb","Mar","Apr","Maí","Jún","Júl","Ágú","Sep","Okt","Nóv","Des"]
counts = np.zeros(12, dtype=int)
for m in months:
    counts[m - 1] += 1

print("\nFjöldi árstoppa eftir mánuðum:")
for i, (n, c) in enumerate(zip(manud_nofn, counts)):
    print(f"  {n}: {c}")

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = ["#2c7bb6" if c == counts.max() else "#abd9e9" for c in counts]
ax.bar(manud_nofn, counts, color=bar_colors, edgecolor="white", linewidth=0.8)
ax.set_title("Fjöldi árstoppa eftir mánuðum – Ölfusá, Selfoss  (1993–2023)")
ax.set_ylabel("Fjöldi topa")
ax.set_xlabel("Mánuður")
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "06a_flod_manadartidni.png", dpi=150)
plt.close()
print("\nMynd vistuð: figures/06a_flod_manadartidni.png")


# ════════════════════════════════════════════════════════════════════════════
# B) LÍKINDADREIFINGARGREINING
# ════════════════════════════════════════════════════════════════════════════

# ── Gringorten plotting positions ────────────────────────────────────────────
#
# Raðar toppum í hækkandi röð og reiknar líkindi:
#
#   F(i) = (i − 0.44) / (N + 0.12)
#
# þar sem i = 1 er lægsti toppurinn.
# Skilar líkindi F ∈ (0,1) og samsvarandi endurkomutíma T = 1/(1−F).

Q_sort  = np.sort(Q_peaks)
ranks   = np.arange(1, N + 1)
F_grng  = (ranks - 0.44) / (N + 0.12)       # Gringorten CDF
T_grng  = 1.0 / (1.0 - F_grng)              # endurkomutími [ár]


# ── Gumbel dreifing ──────────────────────────────────────────────────────────
#
# Gumbel (Extreme Value Type I):
#   F(x) = exp(−exp(−y)),   y = (x − μ) / β
#
# Stikamöt með aðferð minnstu kvaðrata:
#   β  = σ · √6 / π
#   μ  = x̄ − 0.5772 · β      (Euler-Mascheroni fasti ≈ 0.5772)
#
# Bakbreiðing (inverse CDF) gefur:
#   x(T) = μ − β · ln(−ln(1 − 1/T))

mu_g  = Q_sort.mean() - 0.5772 * (Q_sort.std(ddof=1) * np.sqrt(6) / np.pi)
beta_g = Q_sort.std(ddof=1) * np.sqrt(6) / np.pi

def gumbel_ppf(T):
    return mu_g - beta_g * np.log(-np.log(1 - 1 / T))

def gumbel_cdf(x):
    return np.exp(-np.exp(-(x - mu_g) / beta_g))


# ── Log Normal 3 stika (LN3) ─────────────────────────────────────────────────
#
# log(X − ξ) ~ Normal(μ_ln, σ_ln²)
#
# Stikamöt: við finnum neðri mörk ξ sem lágmarkar KS-tölu eða RMSE.
# Kröfur: ξ < min(Q) og ξ > 0 ef við viljum að dreifing gangi á [ξ, ∞).
# Við notum MLE með scipy.stats.lognorm.fit(Q − ξ, floc=0).

def ln3_fit(xi, Q_data):
    """Passar LN3 með gefnu ξ; skilar (sigma, loc, scale) úr scipy."""
    shifted = Q_data - xi
    if np.any(shifted <= 0):
        return None
    return stats.lognorm.fit(shifted, floc=0)

# Leita að besta ξ með lágmörkun RMSE á probability plot
xi_max = Q_sort.min() * 0.999   # ξ verður að vera < min(Q)

def ln3_rmse(xi):
    if xi >= xi_max or xi < 0:
        return 1e9
    params = ln3_fit(xi, Q_sort)
    if params is None:
        return 1e9
    sigma, loc_fit, scale_fit = params
    F_theo = stats.lognorm.cdf(Q_sort - xi, sigma, loc=loc_fit, scale=scale_fit)
    return np.sqrt(np.mean((F_theo - F_grng) ** 2))

res = minimize_scalar(ln3_rmse, bounds=(0, xi_max * 0.99), method="bounded")
xi_ln3 = res.x
sig_ln3, loc_ln3, scale_ln3 = ln3_fit(xi_ln3, Q_sort)

def ln3_ppf(T):
    q_shift = stats.lognorm.ppf(1 - 1/T, sig_ln3, loc=loc_ln3, scale=scale_ln3)
    return q_shift + xi_ln3

def ln3_cdf(x):
    return stats.lognorm.cdf(x - xi_ln3, sig_ln3, loc=loc_ln3, scale=scale_ln3)


# ── Log Pearson 3 stika (LP3) ────────────────────────────────────────────────
#
# y = log₁₀(Q) ~ Pearson Type III dreifing
#
# Stikamöt:
#   ȳ  = meðaltal log-gilda
#   s  = staðalfrávik log-gilda
#   g  = skekja (skewness) log-gilda
#
# scipy.stats.pearson3 með stikana (skew=g, loc=ȳ, scale=s) passar Pearson III
# beint á log-gildin. Bakbreiðing:
#   y(T) = pearson3.ppf(1 − 1/T, g, ȳ, s)
#   Q(T) = 10^y(T)

log_Q  = np.log10(Q_sort)
y_bar  = log_Q.mean()
s_lp3  = log_Q.std(ddof=1)
g_lp3  = stats.skew(log_Q, bias=False)   # sample skewness (bias=False) – nauðsynlegt í LP3

def lp3_ppf(T):
    y_T = stats.pearson3.ppf(1 - 1/T, g_lp3, loc=y_bar, scale=s_lp3)
    return 10 ** y_T

def lp3_cdf(x):
    y = np.log10(np.maximum(x, 1e-9))
    return stats.pearson3.cdf(y, g_lp3, loc=y_bar, scale=s_lp3)


# ════════════════════════════════════════════════════════════════════════════
# VAL Á BESTU DREIFINGU – RMSE á CDF (þyngst eftir efri hala)
# ════════════════════════════════════════════════════════════════════════════
#
# Nota tvo mælikvarða:
#   1. RMSE á öllum gögnum (heildarpassun)
#   2. RMSE á efri 25% (halapassun) – mikilvægast fyrir flóðagreiningu
#
# Besta dreifing er valin sem hefur lægra meðaltal þessara tveggja.

tail_mask = Q_sort >= np.percentile(Q_sort, 75)   # efri 25% toppa

rmse_full = {}
rmse_tail = {}
for name, cdf_fn in [("Gumbel", gumbel_cdf), ("LN3", ln3_cdf), ("LP3", lp3_cdf)]:
    F_theo = cdf_fn(Q_sort)
    rmse_full[name] = np.sqrt(np.mean((F_theo - F_grng) ** 2))
    rmse_tail[name] = np.sqrt(np.mean((F_theo[tail_mask] - F_grng[tail_mask]) ** 2))
    print(f"  RMSE ({name}):  heild={rmse_full[name]:.5f}  hali={rmse_tail[name]:.5f}")

# Sameinaður mælikvarði: meðaltal heildas og hala RMSE
rmse_combined = {n: 0.5 * rmse_full[n] + 0.5 * rmse_tail[n] for n in rmse_full}
best = min(rmse_combined, key=rmse_combined.get)
print(f"\n  Besta dreifing (heild+hali): {best}")


# ════════════════════════════════════════════════════════════════════════════
# Q10, Q50, Q100 – ALLAR ÞRJÁR DREIFINGAR
# ════════════════════════════════════════════════════════════════════════════
Ts = [10, 50, 100]
print("\n══ Endurkomutímar [m³/s] ══")
print(f"{'':12}  {'Q10':>8}  {'Q50':>8}  {'Q100':>8}")
for name, ppf_fn in [("Gumbel", gumbel_ppf), ("LN3", ln3_ppf), ("LP3", lp3_ppf)]:
    vals = [ppf_fn(T) for T in Ts]
    print(f"  {name:10}  {vals[0]:8.1f}  {vals[1]:8.1f}  {vals[2]:8.1f}")


# ════════════════════════════════════════════════════════════════════════════
# 90% ÖRYGGISBIL – BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════
#
# Bootstrap:
#   1. Draga N gildi með endursetningu úr árstoppum (N=30)
#   2. Endurmeta stika valdar dreifingar
#   3. Reikna Q10, Q50, Q100 fyrir hvern endurteiknun
#   4. 5. og 95. hundraðshlutfall = 90% öryggisbil

N_BOOT = 2000

# Nota bestu dreifingu
best_ppf_fn = {"Gumbel": gumbel_ppf, "LN3": ln3_ppf, "LP3": lp3_ppf}[best]

boot_q = {T: [] for T in Ts}

for b in range(N_BOOT):
    Qb   = rng.choice(Q_peaks, size=N, replace=True)
    Qb_s = np.sort(Qb)
    log_Qb = np.log10(np.maximum(Qb_s, 1e-9))

    try:
        if best == "Gumbel":
            mu_b = Qb.mean() - 0.5772 * (Qb.std(ddof=1) * np.sqrt(6) / np.pi)
            bt_b = Qb.std(ddof=1) * np.sqrt(6) / np.pi
            for T in Ts:
                boot_q[T].append(mu_b - bt_b * np.log(-np.log(1 - 1/T)))

        elif best == "LN3":
            # Endurreikna ξ fyrir hverja endurteiknun – lykilatriði fyrir rétt öryggisbil
            xi_max_b = Qb_s.min() * 0.999
            Fg_b = (np.arange(1, N + 1) - 0.44) / (N + 0.12)

            def rmse_b(xi):
                if xi <= 0 or xi >= xi_max_b:
                    return 1e9
                shifted = Qb_s - xi
                if np.any(shifted <= 0):
                    return 1e9
                sg_t, lc_t, sc_t = stats.lognorm.fit(shifted, floc=0)
                Fb = stats.lognorm.cdf(shifted, sg_t, loc=lc_t, scale=sc_t)
                return np.sqrt(np.mean((Fb - Fg_b) ** 2))

            xi_b = minimize_scalar(rmse_b, bounds=(0, xi_max_b * 0.99),
                                   method="bounded").x
            sg, lc, sc = stats.lognorm.fit(Qb_s - xi_b, floc=0)
            for T in Ts:
                boot_q[T].append(stats.lognorm.ppf(1-1/T, sg, lc, sc) + xi_b)

        else:  # LP3
            yb = log_Qb.mean()
            sb = log_Qb.std(ddof=1)
            gb = stats.skew(log_Qb, bias=False)   # sample skewness
            for T in Ts:
                yT = stats.pearson3.ppf(1-1/T, gb, loc=yb, scale=sb)
                boot_q[T].append(10 ** yT)
    except Exception:
        pass

print(f"\n══ 90% öryggisbil (bootstrap, N={N_BOOT}) – {best} ══")
for T in Ts:
    bv = np.array(boot_q[T])
    bv = bv[np.isfinite(bv)]
    lo, hi = np.percentile(bv, [5, 95])
    qval = best_ppf_fn(T)
    print(f"  Q{T:3d} = {qval:7.1f} m³/s   90% ÖB: [{lo:.1f}, {hi:.1f}]")


# ════════════════════════════════════════════════════════════════════════════
# MYNDIR
# ════════════════════════════════════════════════════════════════════════════

# ── Mynd 2: Probability plot – allar þrjár dreifingar ───────────────────────
T_curve = np.logspace(np.log10(1.1), np.log10(500), 300)
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(T_grng, Q_sort, color="k", zorder=5, s=35,
           label="Árstoppar (Gringorten)")

line_styles = ["-", "--", "-."]
cols = ["#2c7bb6", "#d7191c", "#1a9641"]
for (name, ppf_fn), ls, c in zip(
    [("Gumbel", gumbel_ppf), ("LN3", ln3_ppf), ("LP3", lp3_ppf)],
    line_styles, cols
):
    q_curve = [ppf_fn(T) for T in T_curve]
    lbl = f"{name}  ✓" if name == best else name
    ax.plot(T_curve, q_curve, ls, color=c, linewidth=1.8, label=lbl)

# Öryggisbil bestu dreifingar
for T in Ts:
    bv  = np.array(boot_q[T])
    bv  = bv[np.isfinite(bv)]
    lo, hi = np.percentile(bv, [5, 95])
    qval = best_ppf_fn(T)
    ax.errorbar(T, qval, yerr=[[qval - lo], [hi - qval]],
                fmt="none", color="gray", capsize=5, linewidth=1.5, zorder=4)
    ax.annotate(f"Q{T}\n{qval:.0f} m³/s",
                xy=(T, qval), xytext=(T * 1.25, qval + 30),
                fontsize=8, color="gray")

ax.set_xscale("log")
ax.set_xlabel("Endurkomutími T [ár]  (log-skali)", fontsize=11)
ax.set_ylabel("Rennsli Q [m³/s]", fontsize=11)
ax.set_title(f"Flóðagreining – Ölfusá, Selfoss  (1993–2023)\n"
             f"Besta dreifing: {best}  |  Gringorten plotting positions", fontsize=11)
ax.grid(which="major", linestyle="-",  linewidth=0.5, alpha=0.4)
ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "06b_flod_likelihood.png", dpi=150)
plt.close()
print("\nMynd vistuð: figures/06b_flod_likelihood.png")
