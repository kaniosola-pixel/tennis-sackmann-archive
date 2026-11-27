# Projekt SRD: kto wygra mecz ATP?

| Plik | Co zawiera |
|---|---|
| `01_dane_i_cechy.ipynb` / `.html` | Część 1: wczytanie, czyszczenie, losowe A/B, cechy sprzed meczu, testy na wyciek |
| `01_dane_i_cechy.py` | Ten sam notatnik jako skrypt (wygodny do przeglądania zmian w gicie) |
| `dane/atp_cechy_2015_2024.csv` | Gotowy zbiór: 26 759 meczów, cel `A_wins`, kolumna `split` (train 2015–2022 / test 2023–2024) |

Notatnik czyta surowe pliki z `../atp/`, więc uruchamiaj go z folderu `projekt_srd/`.

```python
df = pd.read_csv("dane/atp_cechy_2015_2024.csv", parse_dates=["tourney_date"])
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
```

Proponowane cechy do modeli (`NUM_FEATURES`, `CAT_FEATURES`) są wypisane w punkcie 6 notatnika.
