# %% [markdown]
# # Kto wygra mecz ATP? Część 1: dane i cechy
#
# **Projekt SRD (SGH, zima 2026/27).** Ten notatnik przygotowuje zbiór, na którym pozostali członkowie grupy trenują modele.
#
# Na wyjściu dostajemy plik `dane/atp_cechy_2015_2024.csv`: jeden wiersz to jeden mecz ATP z lat 2015–2024, zmienna celu `A_wins` (1 = wygrał zawodnik A, 0 = wygrał zawodnik B) i cechy policzone **wyłącznie z informacji dostępnych przed meczem**.
#
# Plan notatnika:
# 1. Wczytanie danych (2010–2024; lata 2010–2014 służą tylko jako „rozgrzewka” historii graczy).
# 2. Czyszczenie: kolejność meczów, krecze i walkowery, duplikaty, braki.
# 3. Cechy historyczne w układzie „zawodnik–mecz” (`shift(1)`, czyli tylko przeszłość).
# 4. Ranking Elo liczony sekwencyjnie.
# 5. Losowe przypisanie ról A/B i cechy „A minus B”.
# 6. Imputacja braków, podział czasowy train/test.
# 7. Testy na wyciek i kontrola baseline'u.
# 8. Zapis gotowego zbioru.

# %% [markdown]
# Jeśli brakuje pakietów: `%pip install --upgrade numpy pandas matplotlib seaborn scikit-learn`.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

pd.set_option("display.max_columns", 60)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DATA_DIR = "../atp"            # pliki atp_matches_RRRR.csv z archiwum Sackmanna
FIRST_HISTORY_YEAR = 2010      # od kiedy liczymy historię graczy
FIRST_MODEL_YEAR = 2015        # od kiedy modelujemy
LAST_TRAIN_YEAR = 2022         # trening 2015–2022, test 2023–2024
LAST_YEAR = 2024

# %% [markdown]
# ## 1. Wczytanie danych
#
# Każdy plik to jeden sezon. Łączymy je `pd.concat`. Kolumna `tourney_date` ma format RRRRMMDD i oznacza **poniedziałek rozpoczęcia turnieju**, a nie dzień meczu. Wszystkie mecze turnieju mają więc tę samą datę, co trzeba uwzględnić przy ustalaniu kolejności.

# %%
raw = pd.concat(
    [pd.read_csv(f"{DATA_DIR}/atp_matches_{year}.csv") for year in range(FIRST_HISTORY_YEAR, LAST_YEAR + 1)],
    ignore_index=True,
)
raw["tourney_date"] = pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d")
raw["year"] = raw["tourney_date"].dt.year
print(raw.shape)
raw.head()

# %%
raw.groupby("year").size().to_frame("liczba_meczow").T

# %% [markdown]
# ## 2. Czyszczenie
#
# ### 2.1 Kolejność meczów
#
# Cechy historyczne wymagają poprawnej kolejności. W obrębie turnieju sortujemy po rundzie (od 1. rundy do finału), a w rundzie po `match_num`. `RR` (round robin) to faza grupowa w ATP Finals i Pucharze Davisa, więc stawiamy ją przed półfinałem.

# %%
ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 4.5, "QF": 5, "SF": 6, "BR": 6.5, "F": 7}
raw["round_order"] = raw["round"].map(ROUND_ORDER)
print("Nieznane rundy:", raw["round_order"].isna().sum())

matches = (
    raw.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"])
    .reset_index(drop=True)
)
matches["match_id"] = np.arange(len(matches))   # globalny numer meczu = oś czasu

# %% [markdown]
# ### 2.2 Duplikaty

# %%
print("Duplikaty (tourney_id, match_num):", matches.duplicated(subset=["tourney_id", "match_num"]).sum())

# %% [markdown]
# ### 2.3 Mecze niedokończone
#
# W kolumnie `score` szukamy kreczów (`RET`), walkowerów (`W/O`, `Walkover`), dyskwalifikacji (`DEF`, `Default`) i meczów przerwanych (`ABD`, `abandoned`).
#
# Rozróżniamy dwie rzeczy:
# * **Do modelu** (wiersze zbioru) bierzemy tylko mecze dokończone, bo przy kreczu wynik nie mówi, kto był lepszy.
# * **Do historii gracza** (forma, Elo, H2H) bierzemy wszystkie mecze faktycznie rozegrane, także te z kreczem. Pomijamy tylko walkowery, bo wtedy mecz się nie odbył.

# %%
score = matches["score"].astype(str)
matches["walkover"] = score.str.contains(r"W/O|Walkover", case=False)
matches["unfinished"] = score.str.contains(r"RET|W/O|Walkover|DEF|Def\.|Default|ABD|abandoned", case=False)

in_model_years = matches["year"].between(FIRST_MODEL_YEAR, LAST_YEAR)
print("Mecze 2015–2024:", in_model_years.sum())
print("w tym niedokończone:", (in_model_years & matches["unfinished"]).sum())
print("Zostaje do modelu:", (in_model_years & ~matches["unfinished"]).sum())

# %% [markdown]
# ### 2.4 Braki danych w kolumnach przedmeczowych
#
# Statystyki serwisowe (`w_*`, `l_*`) mają braki głównie w Pucharze Davisa. Imputację zrobimy na końcu, po zbudowaniu cech (punkt 6), żeby mediany liczyć tylko na zbiorze treningowym.

# %%
pre_match_cols = ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
                  "winner_ht", "loser_ht", "winner_age", "loser_age", "surface", "w_svpt", "l_svpt"]
(matches.loc[in_model_years, pre_match_cols].isna().mean() * 100).round(2).to_frame("% braków")

# %% [markdown]
# ### 2.5 Wartości odstające
#
# Wiek i wzrost sprawdzamy regułą IQR (poza przedziałem $[Q_1 - 3 \cdot IQR,\ Q_3 + 3 \cdot IQR]$). Wzrost 3, 15 czy 71 cm to oczywiste błędy wprowadzania, więc zamieniamy je na brak (zostaną zaimputowane w punkcie 6). Ranking i punkty rankingowe są skośne, ale to prawdziwe wartości, więc ich nie usuwamy, tylko logarytmujemy przy budowie cech.

# %%
for col in ["ht", "age"]:
    values = pd.concat([matches[f"winner_{col}"], matches[f"loser_{col}"]])
    q1, q3 = values.quantile([0.25, 0.75])
    low, high = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    for role in ["winner", "loser"]:
        bad = ~matches[f"{role}_{col}"].between(low, high) & matches[f"{role}_{col}"].notna()
        print(f"{role}_{col}: zakres [{low:.1f}, {high:.1f}], poza zakresem: {bad.sum()} -> {sorted(matches.loc[bad, f'{role}_{col}'].unique())}")
        if col == "ht":
            matches.loc[bad, f"{role}_{col}"] = np.nan

# %% [markdown]
# Nawierzchnię `Carpet` (dywan, kilkadziesiąt meczów do 2016 r.) i brak nawierzchni łączymy w kategorię `Other`.

# %%
matches["surface"] = matches["surface"].where(matches["surface"].isin(["Hard", "Clay", "Grass"]), "Other")
matches["surface"].value_counts()

# %% [markdown]
# ## 3. Cechy historyczne: układ „zawodnik–mecz”
#
# Surowy wiersz opisuje mecz z perspektywy zwycięzcy i przegranego. Żeby policzyć historię **jednego zawodnika**, rozkładamy każdy mecz na dwa wiersze: jeden dla zwycięzcy (`win = 1`), drugi dla przegranego (`win = 0`). Wtedy wystarczy `groupby("player_id")`.
#
# Przy okazji liczymy statystyki z danego meczu, które **same w sobie są wyciekiem**, ale ich średnia z **poprzednich** meczów jest legalną cechą:
# * `serve_won` = odsetek wygranych punktów przy własnym serwisie,
# * `return_won` = odsetek wygranych punktów przy serwisie rywala.

# %%
played = matches[~matches["walkover"]].copy()   # historia: wszystkie rozegrane mecze


def player_rows(df, me, opp, win):
    """Jeden wiersz na zawodnika `me` w każdym meczu (me/opp to prefiksy 'winner'/'loser')."""
    s_me, s_opp = ("w", "l") if me == "winner" else ("l", "w")
    out = pd.DataFrame({
        "match_id": df["match_id"],
        "tourney_id": df["tourney_id"],
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "player_id": df[f"{me}_id"],
        "opp_id": df[f"{opp}_id"],
        "win": win,
        "serve_won": (df[f"{s_me}_1stWon"] + df[f"{s_me}_2ndWon"]) / df[f"{s_me}_svpt"],
        "return_won": 1 - (df[f"{s_opp}_1stWon"] + df[f"{s_opp}_2ndWon"]) / df[f"{s_opp}_svpt"],
    })
    return out


long = pd.concat([player_rows(played, "winner", "loser", 1),
                  player_rows(played, "loser", "winner", 0)])
long = long.replace([np.inf, -np.inf], np.nan)          # svpt = 0 daje dzielenie przez zero
long = long.sort_values(["player_id", "match_id"]).reset_index(drop=True)
print(long.shape)
long.head()

# %% [markdown]
# **Kluczowy trik: `shift(1)`.** W każdej grupie (zawodnik) przesuwamy kolumnę o jeden wiersz w dół. Wartość w wierszu meczu *t* pochodzi wtedy z meczu *t − 1*, więc wynik bieżącego meczu nigdy nie trafia do jego własnej cechy. Dopiero na przesuniętej kolumnie liczymy `rolling` (średnia krocząca) albo `cumsum`.
#
# Cechy:
# * `n_prev` – liczba wcześniejszych meczów (od 2010 r.), czyli miara „zimnego startu”,
# * `form10` – odsetek wygranych w ostatnich 10 meczach,
# * `surface_wr` – odsetek wygranych w ostatnich 20 meczach na **tej** nawierzchni, wygładzony w stronę 50% (prior o wadze 5 meczów), żeby gracz z bilansem 1–0 nie dostał 100%,
# * `h2h_wins`, `h2h_n` – wygrane i liczba wcześniejszych meczów z tym samym rywalem,
# * `days_since_last_tourney` – przerwa od poprzedniego turnieju (w dniach),
# * `matches_14d` – liczba meczów w ostatnich 14 dniach (łącznie z wcześniejszymi rundami bieżącego turnieju),
# * `serve_avg`, `return_avg` – średnie z ostatnich 20 meczów ze statystykami.

# %%
g = long.groupby("player_id")

long["n_prev"] = g.cumcount()
long["form10"] = g["win"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
long["serve_avg"] = g["serve_won"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
long["return_avg"] = g["return_won"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())

# odsetek wygranych na nawierzchni, wygładzony: (wygrane + 5 * 0.5) / (mecze + 5)
gs = long.groupby(["player_id", "surface"])["win"]
PRIOR_N = 5
surf_wins = gs.transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum()).fillna(0)
surf_n = gs.transform(lambda s: s.shift(1).rolling(20, min_periods=1).count()).fillna(0)
long["surface_wr"] = (surf_wins + PRIOR_N * 0.5) / (surf_n + PRIOR_N)
long["surface_n_prev"] = long.groupby(["player_id", "surface"]).cumcount()

# head-to-head: historia pary (zawodnik, rywal)
gh = long.groupby(["player_id", "opp_id"])["win"]
long["h2h_wins"] = gh.transform(lambda s: s.shift(1).cumsum()).fillna(0)
long["h2h_n"] = long.groupby(["player_id", "opp_id"]).cumcount()

# %% [markdown]
# Przerwa od poprzedniego turnieju: bierzemy jeden wiersz na (zawodnik, turniej) i liczymy różnicę dat startu kolejnych turniejów. Brak poprzedniego turnieju (pierwszy mecz w danych) uzupełniamy na razie wartością NaN.
#
# Liczba meczów w 14 dniach: dla każdego meczu liczymy, ile **wcześniejszych** meczów zawodnika ma datę turnieju nie starszą niż 14 dni. `np.searchsorted` znajduje w posortowanych datach pierwszy mecz w oknie, a różnica pozycji daje liczbę meczów.

# %%
tour = long.drop_duplicates(["player_id", "tourney_id"])[["player_id", "tourney_id", "tourney_date"]].copy()
tour["days_since_last_tourney"] = tour.groupby("player_id")["tourney_date"].diff().dt.days
long = long.merge(tour[["player_id", "tourney_id", "days_since_last_tourney"]], on=["player_id", "tourney_id"], how="left")


def matches_last_14d(dates):
    d = dates.values
    first_in_window = np.searchsorted(d, d - np.timedelta64(14, "D"), side="left")
    return pd.Series(np.arange(len(d)) - first_in_window, index=dates.index)


long = long.sort_values(["player_id", "match_id"]).reset_index(drop=True)
long["matches_14d"] = long.groupby("player_id", group_keys=False)["tourney_date"].apply(matches_last_14d)
long[["n_prev", "form10", "surface_wr", "h2h_wins", "h2h_n", "days_since_last_tourney", "matches_14d", "serve_avg", "return_avg"]].describe().T.round(3)

# %% [markdown]
# ## 4. Ranking Elo
#
# Elo to alternatywa dla rankingu ATP: każdy gracz startuje z 1500 punktami, a po każdym meczu zwycięzca zabiera przegranemu część punktów, tym więcej, im bardziej wygrana była niespodziewana. Oczekiwane prawdopodobieństwo wygranej A:
#
# $$P(A) = \frac{1}{1 + 10^{(Elo_B - Elo_A)/400}}$$
#
# Współczynnik K maleje z liczbą rozegranych meczów (wariant FiveThirtyEight: $K = 250 / (n + 5)^{0.4}$), więc nowi gracze szybko „znajdują” swój poziom, a doświadczeni są stabilniejsi. Liczymy też osobne Elo dla każdej nawierzchni.
#
# Do cechy zapisujemy Elo **przed** meczem, a dopiero potem je aktualizujemy. Lata 2010–2014 to rozgrzewka, żeby w 2015 r. wartości były już stabilne.

# %%
def compute_elo(df):
    elo, elo_surf = {}, {}
    n_played, n_played_surf = {}, {}
    rows = []
    for m in df.itertuples(index=False):
        w, l, s = m.winner_id, m.loser_id, m.surface
        ew, el = elo.get(w, 1500.0), elo.get(l, 1500.0)
        sw, sl = elo_surf.get((w, s), 1500.0), elo_surf.get((l, s), 1500.0)
        rows.append((m.match_id, ew, el, sw, sl))       # zapis PRZED aktualizacją

        p_w = 1 / (1 + 10 ** ((el - ew) / 400))
        k_w = 250 / (n_played.get(w, 0) + 5) ** 0.4
        k_l = 250 / (n_played.get(l, 0) + 5) ** 0.4
        elo[w], elo[l] = ew + k_w * (1 - p_w), el - k_l * (1 - p_w)

        ps_w = 1 / (1 + 10 ** ((sl - sw) / 400))
        ks_w = 250 / (n_played_surf.get((w, s), 0) + 5) ** 0.4
        ks_l = 250 / (n_played_surf.get((l, s), 0) + 5) ** 0.4
        elo_surf[(w, s)], elo_surf[(l, s)] = sw + ks_w * (1 - ps_w), sl - ks_l * (1 - ps_w)

        n_played[w], n_played[l] = n_played.get(w, 0) + 1, n_played.get(l, 0) + 1
        n_played_surf[(w, s)] = n_played_surf.get((w, s), 0) + 1
        n_played_surf[(l, s)] = n_played_surf.get((l, s), 0) + 1
    return pd.DataFrame(rows, columns=["match_id", "winner_elo", "loser_elo", "winner_surface_elo", "loser_surface_elo"])


elo = compute_elo(played.sort_values("match_id"))
elo.tail()

# %% [markdown]
# ## 5. Losowe role A/B i cechy „A minus B”
#
# W surowych danych zmienna celu jest zaszyta w nazwach kolumn (`winner_*` zawsze wygrywa). Gdybyśmy zostawili tę strukturę, model nauczyłby się „kolumna winner wygrywa” i miałby 100% trafności. Dlatego dla każdego meczu losujemy (`np.random.default_rng(42)`), czy zwycięzca zostaje zawodnikiem A, czy B. Klasy są wtedy zbalansowane ok. 50/50.
#
# Najpierw dołączamy cechy historyczne zwycięzcy i przegranego, potem robimy zamianę.

# %%
HIST_COLS = ["n_prev", "form10", "surface_wr", "surface_n_prev", "h2h_wins", "h2h_n",
             "days_since_last_tourney", "matches_14d", "serve_avg", "return_avg"]

df = matches[in_model_years & ~matches["unfinished"]].copy()
for role in ["winner", "loser"]:
    feats = long[["match_id", "player_id"] + HIST_COLS].rename(columns={c: f"{role}_{c}" for c in HIST_COLS})
    df = df.merge(feats, left_on=["match_id", f"{role}_id"], right_on=["match_id", "player_id"], how="left").drop(columns="player_id")
df = df.merge(elo, on="match_id", how="left")
print(df.shape)

# %%
rng = np.random.default_rng(RANDOM_STATE)
winner_is_A = rng.random(len(df)) < 0.5
df["A_wins"] = winner_is_A.astype(int)

PLAYER_COLS = ["id", "name", "hand", "ht", "age", "rank", "rank_points", "seed", "entry",
               "elo", "surface_elo"] + HIST_COLS
for c in PLAYER_COLS:
    df[f"A_{c}"] = np.where(winner_is_A, df[f"winner_{c}"], df[f"loser_{c}"])
    df[f"B_{c}"] = np.where(winner_is_A, df[f"loser_{c}"], df[f"winner_{c}"])

# kolumny z oryginalnymi nazwami winner/loser oraz wynik meczu usuwamy (wyciek)
leak_cols = [c for c in df.columns if c.startswith(("winner_", "loser_", "w_", "l_"))] + ["score", "minutes"]
df = df.drop(columns=leak_cols)
df["A_wins"].value_counts(normalize=True).round(3)

# %% [markdown]
# Kolumny po `np.where` mają typ `object`, więc przywracamy typy liczbowe.

# %%
num_player_cols = ["ht", "age", "rank", "rank_points", "seed", "elo", "surface_elo"] + HIST_COLS
for p in ["A", "B"]:
    for c in num_player_cols:
        df[f"{p}_{c}"] = pd.to_numeric(df[f"{p}_{c}"])
    df[f"{p}_id"] = df[f"{p}_id"].astype(int)

# %% [markdown]
# ## 6. Braki danych, cechy porównawcze i podział czasowy
#
# Podział: **trening 2015–2022, test 2023–2024**. Wszystkie wartości do imputacji (mediany) liczymy tylko na treningu.
#
# Zasady imputacji (każdy brak dostaje flagę, bo brak to też informacja):
# * brak rankingu = gracz spoza rankingu → ranking 2000, punkty 0, flaga `*_unranked`,
# * brak wzrostu → mediana z treningu, flaga `*_ht_missing`,
# * brak wieku → mediana z treningu,
# * brak średnich serwisowych (za mało meczów ze statystykami) → mediana z treningu,
# * brak przerwy (pierwszy turniej w danych) → 28 dni,
# * ręka: `U` (nieznana) i `A` (oburęczny, 1 przypadek) łączymy w `U`.

# %%
df["split"] = np.where(df["year"] <= LAST_TRAIN_YEAR, "train", "test")
is_train = df["split"] == "train"

for p in ["A", "B"]:
    df[f"{p}_unranked"] = df[f"{p}_rank"].isna().astype(int)
    df[f"{p}_ht_missing"] = df[f"{p}_ht"].isna().astype(int)
    df[f"{p}_hand"] = df[f"{p}_hand"].where(df[f"{p}_hand"].isin(["R", "L"]), "U")
    df[f"{p}_seeded"] = df[f"{p}_seed"].notna().astype(int)
    df[f"{p}_qualifier"] = df[f"{p}_entry"].isin(["Q", "LL"]).astype(int)
    df[f"{p}_wildcard"] = df[f"{p}_entry"].eq("WC").astype(int)

df[["A_rank", "B_rank"]] = df[["A_rank", "B_rank"]].fillna(2000)
df[["A_rank_points", "B_rank_points"]] = df[["A_rank_points", "B_rank_points"]].fillna(0)
df[["A_days_since_last_tourney", "B_days_since_last_tourney"]] = df[["A_days_since_last_tourney", "B_days_since_last_tourney"]].fillna(28)

for c in ["ht", "age", "serve_avg", "return_avg"]:
    train_median = pd.concat([df.loc[is_train, f"A_{c}"], df.loc[is_train, f"B_{c}"]]).median()
    df[[f"A_{c}", f"B_{c}"]] = df[[f"A_{c}", f"B_{c}"]].fillna(train_median)
    print(f"{c}: mediana z treningu = {train_median:.3f}")

# %% [markdown]
# Cechy porównawcze liczymy jako **różnicę A minus B**. Gdy zamienimy graczy miejscami, różnica zmienia znak, a cel się odwraca, więc model „widzi” mecz symetrycznie. Ranking i punkty są silnie skośne (ranking do ponad 2000), więc bierzemy różnicę logarytmów.

# %%
df["rank_diff"] = df["A_rank"] - df["B_rank"]
df["log_rank_diff"] = np.log(df["A_rank"]) - np.log(df["B_rank"])
df["points_diff"] = np.log1p(df["A_rank_points"]) - np.log1p(df["B_rank_points"])
df["age_diff"] = df["A_age"] - df["B_age"]
df["height_diff"] = df["A_ht"] - df["B_ht"]
df["elo_diff"] = df["A_elo"] - df["B_elo"]
df["surface_elo_diff"] = df["A_surface_elo"] - df["B_surface_elo"]
df["form_diff"] = df["A_form10"].fillna(0.5) - df["B_form10"].fillna(0.5)
df["surface_winrate_diff"] = df["A_surface_wr"] - df["B_surface_wr"]
df["h2h_diff"] = df["A_h2h_wins"] - df["B_h2h_wins"]
df["h2h_n"] = df["A_h2h_n"]
df["rest_days_diff"] = df["A_days_since_last_tourney"].clip(upper=365) - df["B_days_since_last_tourney"].clip(upper=365)
df["matches_14d_diff"] = df["A_matches_14d"] - df["B_matches_14d"]
df["serve_diff"] = df["A_serve_avg"] - df["B_serve_avg"]
df["return_diff"] = df["A_return_avg"] - df["B_return_avg"]
df["log_n_prev_diff"] = np.log1p(df["A_n_prev"]) - np.log1p(df["B_n_prev"])
df["lefty_vs_righty"] = ((df["A_hand"] == "L") ^ (df["B_hand"] == "L")).astype(int)
df["seeded_diff"] = df["A_seeded"] - df["B_seeded"]
df["davis_cup"] = (df["tourney_level"] == "D").astype(int)

# %% [markdown]
# Lista cech zalecanych do modeli (pozostałe kolumny, np. `A_name`, `tourney_name`, zostają w pliku tylko do opisu i analizy segmentów).

# %%
NUM_FEATURES = [
    "log_rank_diff", "points_diff", "elo_diff", "surface_elo_diff",
    "age_diff", "height_diff", "form_diff", "surface_winrate_diff",
    "h2h_diff", "h2h_n", "rest_days_diff", "matches_14d_diff",
    "serve_diff", "return_diff", "log_n_prev_diff", "seeded_diff",
    "A_unranked", "B_unranked", "A_ht_missing", "B_ht_missing",
    "A_qualifier", "B_qualifier", "A_wildcard", "B_wildcard",
    "lefty_vs_righty", "best_of",
]
CAT_FEATURES = ["surface", "tourney_level", "round", "A_hand", "B_hand"]
TARGET = "A_wins"

print(len(NUM_FEATURES), "cech liczbowych +", len(CAT_FEATURES), "kategorycznych")
print("Braki w cechach:", df[NUM_FEATURES + CAT_FEATURES].isna().sum().sum())
df[NUM_FEATURES].describe().T.round(3)

# %%
df["split"].value_counts()

# %% [markdown]
# ## 7. Kontrole jakości i testy na wyciek
#
# ### 7.1 Balans klas i baseline
#
# Baseline „wygrywa zawodnik z lepszym rankingiem” to próg, który modele muszą pobić. Dla porównania liczymy też regułę „wygrywa zawodnik z wyższym Elo”.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x="A_wins", hue="split", ax=ax[0])
ax[0].set_title("Balans klas A_wins")

year_acc = (
    df.assign(rank_rule=(df["A_rank"] < df["B_rank"]).astype(int) == df["A_wins"],
              elo_rule=(df["elo_diff"] > 0).astype(int) == df["A_wins"])
    .groupby("year")[["rank_rule", "elo_rule"]].mean()
)
year_acc.plot(marker="o", ax=ax[1])
ax[1].set_title("Trafność reguł w kolejnych latach")
ax[1].set_ylabel("accuracy")
plt.tight_layout()
plt.show()

# %%
def rule_accuracy(mask):
    d = df[mask]
    return pd.Series({
        "lepszy ranking": ((d["A_rank"] < d["B_rank"]).astype(int) == d["A_wins"]).mean(),
        "wyższe Elo": ((d["elo_diff"] > 0).astype(int) == d["A_wins"]).mean(),
        "liczba meczów": len(d),
    })


pd.DataFrame({"całość": rule_accuracy(df["year"] > 0),
              "train": rule_accuracy(is_train),
              "test": rule_accuracy(~is_train)}).round(3)

# %% [markdown]
# ### 7.2 Test symetrii
#
# Jeśli zamiana A↔B była poprawna, to po odwróceniu ról różnice zmieniają znak, a cel się odwraca. Sprawdzamy też, że po losowaniu nie zostało żadne powiązanie roli z wynikiem: średnia `log_rank_diff` w grupie `A_wins = 1` musi być ujemna (A ma lepszy ranking), a w grupie `A_wins = 0` dodatnia o podobnej wielkości.

# %%
df.groupby("A_wins")[["log_rank_diff", "elo_diff", "form_diff", "h2h_diff"]].mean().round(3)

# %% [markdown]
# ### 7.3 Test „ślepoty na przyszłość”
#
# Najmocniejszy test na wyciek czasowy: liczymy cechy historyczne jeszcze raz, ale **tylko na danych do końca 2019 r.** Jeżeli jakaś cecha meczu z 2019 r. korzysta z przyszłości, to jej wartość się zmieni, bo przyszłych meczów nie ma w danych. Wszystkie wartości muszą być identyczne.

# %%
def history_features(played_df):
    """Ta sama logika co w punkcie 3, zamknięta w funkcji."""
    lg = pd.concat([player_rows(played_df, "winner", "loser", 1), player_rows(played_df, "loser", "winner", 0)])
    lg = lg.replace([np.inf, -np.inf], np.nan).sort_values(["player_id", "match_id"]).reset_index(drop=True)
    g = lg.groupby("player_id")
    lg["n_prev"] = g.cumcount()
    lg["form10"] = g["win"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    lg["serve_avg"] = g["serve_won"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
    gs = lg.groupby(["player_id", "surface"])["win"]
    sw = gs.transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum()).fillna(0)
    sn = gs.transform(lambda s: s.shift(1).rolling(20, min_periods=1).count()).fillna(0)
    lg["surface_wr"] = (sw + PRIOR_N * 0.5) / (sn + PRIOR_N)
    lg["h2h_wins"] = lg.groupby(["player_id", "opp_id"])["win"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    lg["matches_14d"] = lg.groupby("player_id", group_keys=False)["tourney_date"].apply(matches_last_14d)
    return lg


check_cols = ["n_prev", "form10", "serve_avg", "surface_wr", "h2h_wins", "matches_14d"]
cut = played[played["year"] <= 2019]
full_2019 = long[long["tourney_date"].dt.year == 2019].set_index(["match_id", "player_id"])[check_cols]
cut_2019 = history_features(cut)
cut_2019 = cut_2019[cut_2019["tourney_date"].dt.year == 2019].set_index(["match_id", "player_id"])[check_cols]

elo_cut = compute_elo(cut.sort_values("match_id")).set_index("match_id")
elo_full = elo.set_index("match_id").loc[elo_cut.index]

pd.Series({
    "cechy historyczne identyczne": full_2019.sort_index().equals(cut_2019.sort_index()),
    "Elo identyczne": np.allclose(elo_full.values, elo_cut.values),
})

# %% [markdown]
# ### 7.4 Szybki test „czy wynik nie jest za dobry”
#
# To nie jest jeszcze modelowanie (to część 2 projektu), tylko kontrola: prosta regresja logistyczna na cechach liczbowych, uczona na 2015–2022 i sprawdzona na 2023–2024. Wynik w okolicach 65–70% jest realistyczny. Wynik powyżej 80% oznaczałby, że do cech przeciekła informacja o wyniku.

# %%
X_train, y_train = df.loc[is_train, NUM_FEATURES], df.loc[is_train, TARGET]
X_test, y_test = df.loc[~is_train, NUM_FEATURES], df.loc[~is_train, TARGET]

sanity = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
sanity.fit(X_train, y_train)
proba = sanity.predict_proba(X_test)[:, 1]
print(f"Test 2023–2024: accuracy = {accuracy_score(y_test, proba > 0.5):.3f}, AUC = {roc_auc_score(y_test, proba):.3f}")

# %%
coefs = pd.Series(sanity[-1].coef_[0], index=NUM_FEATURES).sort_values()
coefs.plot.barh(figsize=(7, 7), title="Współczynniki regresji logistycznej (cechy wystandaryzowane)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Zapis gotowego zbioru
#
# Zapisujemy identyfikatory (do analizy segmentów i podglądu), kontekst meczu, surowe cechy A/B, cechy porównawcze, cel i kolumnę `split`. Pozostałe części projektu wczytują plik i od razu dzielą go po `split`.

# %%
ID_COLS = ["match_id", "tourney_id", "tourney_name", "tourney_date", "year", "split",
           "A_id", "A_name", "B_id", "B_name"]
CONTEXT_COLS = ["surface", "tourney_level", "round", "best_of", "draw_size", "davis_cup"]
RAW_AB = [f"{p}_{c}" for c in ["rank", "rank_points", "age", "ht", "hand", "elo", "surface_elo", "form10",
                                "surface_wr", "n_prev", "matches_14d", "days_since_last_tourney",
                                "serve_avg", "return_avg"] for p in ["A", "B"]]
DIFF_COLS = ["rank_diff", "log_rank_diff", "points_diff", "elo_diff", "surface_elo_diff", "age_diff",
             "height_diff", "form_diff", "surface_winrate_diff", "h2h_diff", "h2h_n", "rest_days_diff",
             "matches_14d_diff", "serve_diff", "return_diff", "log_n_prev_diff", "seeded_diff", "lefty_vs_righty"]
FLAG_COLS = [f"{p}_{c}" for c in ["unranked", "ht_missing", "seeded", "qualifier", "wildcard"] for p in ["A", "B"]]

out_cols = list(dict.fromkeys(ID_COLS + CONTEXT_COLS + RAW_AB + DIFF_COLS + FLAG_COLS + [TARGET]))
final = df[out_cols].sort_values("match_id").reset_index(drop=True)
final.to_csv("dane/atp_cechy_2015_2024.csv", index=False, float_format="%.5g")
print(final.shape)
final.head()

# %% [markdown]
# ### Jak wczytać zbiór w kolejnych notatnikach
#
# ```python
# df = pd.read_csv("dane/atp_cechy_2015_2024.csv", parse_dates=["tourney_date"])
# train, test = df[df["split"] == "train"], df[df["split"] == "test"]
# ```
#
# Kolumny `NUM_FEATURES` i `CAT_FEATURES` z punktu 6 to proponowany zestaw cech. Do `TimeSeriesSplit` zbiór jest już posortowany chronologicznie (`match_id`).
