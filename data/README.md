# Datasets

All raw files live in [`raw/`](raw/) and are loaded **unmodified** through `mlaz` (e.g. `mlaz.load_titanic()`),
so every cleaning step is visible in the notebooks. The notebooks never download anything.

| File | Rows × cols | Target | Used in | Source & licence |
|---|---|---|---|---|
| `titanic.csv` | 891 × 15 | `survived` (binary) | 01, 02, 04, 11 | Titanic passenger list, via [seaborn-data](https://github.com/mwaskom/seaborn-data) (BSD-3) |
| `mpg.csv` | 398 × 9 | `mpg` (regression) | 02, 03, 04, 05, 11 | UCI Auto MPG (Quinlan, 1993), via seaborn-data |
| `penguins.csv` | 344 × 7 | `species` (3 classes) | 03, 06, 10 | Palmer Penguins (Gorman, Williams & Fraser, 2014; CC0), via seaborn-data |
| `concrete.csv` | 1030 × 9 | `strength` in MPa (regression) | 12 | UCI Concrete Compressive Strength, I-C. Yeh (1998), CC BY 4.0 |

`concrete.csv` columns: `cement`, `slag` (blast-furnace slag), `ash` (fly ash), `water`, `superplastic`
(superplasticizer), `coarseagg`, `fineagg` — all in kg per m³ of concrete — plus `age` (days) and `strength` (MPa).

Chapters also use scikit-learn's bundled datasets (`load_iris`, `load_wine`, `load_breast_cancer`, `load_diabetes`,
`load_digits`) and synthetic generators (`make_moons`, `make_classification`, …).
