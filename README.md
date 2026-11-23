# Hydrogen Adsorption Prediction in Metal–Organic Frameworks Using Machine Learning

Code, cleaned dataset, trained models and results for predicting hydrogen (H₂) storage in metal–organic frameworks (MOFs) from structure-derived descriptors.

**Authors:** Vishal; Piyush Kumar Maurya
**Supervisor:** Prof. Vimal Chandra Srivastava
**Affiliation:** Department of Chemical Engineering, Indian Institute of Technology Roorkee, India
**Version:** 1.0.0 (November 2026)

## Summary

10,000 hypothetical MOFs (hMOFs) were downloaded from MOFX-DB with their crystal structures (CIF) and simulated H₂ isotherms at 77 K (2 and 100 bar). We computed 36 geometric, structural and compositional descriptors per MOF. After cleaning, 9,958 MOFs remained. Five regression models were trained: Linear, Ridge, Random Forest, Histogram Gradient Boosting and XGBoost. They predict six targets:

| Target | Meaning | Unit |
|---|---|---|
| `cap_gL`, `cap_wt` | Total H₂ uptake at 77 K, 100 bar | g/L, wt% |
| `del_ps_gL`, `del_ps_wt` | Deliverable capacity at 77 K, 100 → 2 bar | g/L, wt% |
| `uptake_gL`, `uptake_mgg` | Uptake at 2 or 100 bar (pressure is a model input) | g/L, mg/g |

The test set holds 20% of MOFs, split so that each MOFkey falls entirely in train or test. Model selection used 5-fold grouped cross-validation on the training set.

**Best model per target (test set):**

| Target | Best model | Test R² | Test MAE | Test RMSE | CV R² |
|---|---|---|---|---|---|
| cap_gL | Hist. Gradient Boosting | 0.988 | 0.87 g/L | 1.28 g/L | 0.988 |
| cap_wt | XGBoost | 0.999 | 0.091 wt% | 0.120 wt% | 0.999 |
| del_ps_gL | XGBoost | 0.994 | 0.81 g/L | 1.14 g/L | 0.993 |
| del_ps_wt | XGBoost | 0.999 | 0.098 wt% | 0.129 wt% | 0.999 |
| uptake_gL | XGBoost | 0.994 | 0.88 g/L | 1.26 g/L | 0.994 |
| uptake_mgg | XGBoost | 0.999 | 1.00 mg/g | 1.33 mg/g | 0.999 |

## Contents

```
code/
  mof_h2_colab.py              Full pipeline: fetch → features → cleaning → training → results → screening → checks
  mof_h2_full_pipeline.ipynb   Same pipeline as a Google Colab notebook
data/
  ml_wide_clean.csv            9,958 MOFs × (36 features + targets), one row per MOF
  ml_long_clean.csv            19,916 rows, one row per MOF and pressure (2 and 100 bar)
  data_dictionary.csv          Description, unit and role of every column
results/
  tables/                      Metrics of all models, best models, cleaning report, permutation
                               importance, test-set predictions, feature-set baseline,
                               extrapolation test, screening predictions
  figures/                     Parity, residual, learning-curve, importance and feature plots;
                               model comparison; correlation heatmap; target distributions
models/
  <target>.joblib              Best trained model for each of the six targets
requirements.txt               Package versions used to train the saved models
CITATION.cff                   How to cite this record
LICENSE                        MIT licence (code)
LICENSE-DATA.md                CC BY 4.0 licence (data, results, figures, models)
```

## Reproducing the results

1. Install the packages: `pip install -r requirements.txt`. In Google Colab, open `code/mof_h2_full_pipeline.ipynb` and run all cells.
2. Run `python code/mof_h2_colab.py`. It downloads the structures and isotherms from MOFX-DB, which takes some time. It then builds the features, cleans the data, trains all models and writes `results/` and `models/`.
3. All settings (number of MOFs, pressures, test fraction, random seed = 42) are at the top of the script.

CIF files and raw isotherms are not included here. They belong to MOFX-DB and are downloaded again by the script.

## Using a saved model

```python
import joblib, pandas as pd

bundle = joblib.load("models/del_ps_gL.joblib")
df = pd.read_csv("data/ml_wide_clean.csv")
pred = bundle["model"].predict(df[bundle["features"]])
print(bundle["model_name"], bundle["metrics"]["test_r2"])
```

Each `.joblib` file is a dictionary holding:
- `model`: the fitted model
- `model_name`
- `features`: the exact input columns, in order
- `target`
- `metrics`: test and cross-validation scores

To predict for a new MOF, compute the same descriptors with `build_features()` in the script.

**Version note:** the models were saved with scikit-learn 1.9.1 and XGBoost 3.4.1 (Python 3.13). Load them with the same versions, or retrain with the script.

## Notes and limitations

- **Single metal.** All MOFs in this subset (the first 10,000 hMOFs) are Zn-based, so the effect of the metal cannot be learned.
- **Limited conditions.** Only 77 K, at 2 and 100 bar, was available. The uptake models should not be used at other conditions.
- **Simulated data.** All targets are GCMC simulations: rigid frameworks, Darkrim–Levesque model for H₂, UFF for the framework. They are not experimental measurements.
- **No deduplication.** Structures sharing a MOFkey were kept, because they differ in interpenetration or substituent positions. MOFkey is used only to group the train/test split.

## Data source and attribution

Structures and adsorption data come from MOFX-DB (https://mof.tech.northwestern.edu). Please also cite:

Bobbitt, N. S.; Shi, K.; Bucior, B. J.; Chen, H.; Tracy-Amoroso, N.; Li, Z.; Sun, Y.; Merlin, J. H.; Siepmann, J. I.; Siderius, D. W.; Snurr, R. Q. MOFX-DB: An Online Database of Computational Adsorption Data for Nanoporous Materials. *J. Chem. Eng. Data* **2023**, *68* (2), 483–498. https://doi.org/10.1021/acs.jced.2c00583

Wilmer, C. E.; Leaf, M.; Lee, C. Y.; Farha, O. K.; Hauser, B. G.; Hupp, J. T.; Snurr, R. Q. Large-Scale Screening of Hypothetical Metal–Organic Frameworks. *Nat. Chem.* **2012**, *4*, 83–89. https://doi.org/10.1038/nchem.1192

## Licence

- **Code** (`code/`): MIT licence; see `LICENSE`.
- **Data, results, figures and models:** CC BY 4.0; see `LICENSE-DATA.md`.

## Citation

See `CITATION.cff`, or cite this Zenodo record by its DOI.
