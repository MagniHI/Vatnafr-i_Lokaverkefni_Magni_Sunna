# Lokaverkefni í Vatnafræði – Ölfusá við Selfoss
**Höfundar:** Magni Snær & Sunna

Kóðar sem notaðir eru til að vinna gögn og framleiða myndir fyrir lokaverkefni í vatnafræði. Greiningin byggir á gögnum frá LamaH-Ice fyrir Ölfusá við Selfoss (stöð ID 98) á tímabilinu **1993-10-01 – 2023-09-30**.

---

## Uppbygging verkefnisins

```
VATNLOKA/
├── data/                  # Inntaksgögn og unnin gögn
│   ├── olfus_flaedi_93_23.csv       # Dagleg rennslismæling [m³/s]
│   ├── olfus_vedur_93_23.csv        # Dagleg veðurgögn (úrkoma, hitastig)
│   ├── olfus_eiginleikar.csv        # Eiginleikar vatnasviðsins
│   ├── recession_params.json        # Samdráttarstuðlar (k) frá skref 3
│   └── ...                          # Upprunalegar LamaH-Ice skrár
├── figures/               # Myndir sem kóðar framleiða
├── gis/                   # GIS gögn
└── repo/
    └── scripts/           # Python skriptur (skref 00–08)
```

---

## Skriptur

Keyrðar í röð:

| Skref | Skrá | Lýsing |
|-------|------|---------|
| 00 | `00_hreinsa_gogn.py` | Hleður hrágögnum LamaH-Ice, síar á tímabilið og flytur út hreinar CSV skrár |
| 01 | `01_lysing_vatnasviðs.py` | Sækir eiginleika vatnasviðsins og framleiðir lýsandi tölfræði |
| 02 | `02_arstidarsveifla.py` | Reiknar meðaltalsár (climatology) fyrir rennsli, úrkomu og hitastig |
| 03 | `03_grunnvatnsframlag.py` | Mat á grunnvatnsframlagi með Lyne-Hollick síu (BFI) |
| 04 | `04_grunnliking.py` | Tengir gögn við grunnlíkingu vatnafræðinnar |
| 05 | `05_langaeslina.py` | Langæislína rennslis (Flow Duration Curve) |
| 06 | `06_flodagreining.py` | Flóðagreining – flóðatíðni eftir mánuðum og líkindadreifingargreining |
| 07 | `07_leitnigreining.py` | Leitnigreining með Theil-Sen og Mann-Kendall prófi |
| 08 | `08_rennslisatburdur.py` | Greining á stærsta flóðatburði (hlýviðurflóð, 21. des 2006, Q = 1931 m³/s) |

---

## Helstu niðurstöður

- **BFI = 0.80** – hátt grunnvatnsframlag, einkennandi fyrir gljúpur jarðveg á vatnasviðinu
- **Stærsta flóð:** 1931 m³/s þann 21. desember 2006 (hlýviðurflóð, T ≈ 80–100 ár)
- **Leitni:** [sjá skref 07]
- **Samdráttarstuðull:** k ≈ 0.029 dag⁻¹ (τ ≈ 34.6 dagar)

---

## Keyrsluumhverfi

```bash
pip install numpy pandas matplotlib scipy
```

Keyrsla frá rótarskrá verkefnisins:

```bash
python repo/scripts/00_hreinsa_gogn.py
python repo/scripts/01_lysing_vatnasviðs.py
# ... o.s.frv. í röð
```

---

## Gagnaveita

Gögn fengin frá **LamaH-Ice** (Large-Sample Hydrology fyrir Ísland).
