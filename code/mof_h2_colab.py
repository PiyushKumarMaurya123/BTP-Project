import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mofdb_client import fetch
from pymatgen.core import Structure
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")

DATABASE = "hMOF"
LIMIT = 10000
GAS = "H2"
LOADING_UNIT = "mg/g"
PRESSURE_UNIT = "bar"
REFETCH = False
DEDUP_BY = None
SAVE_CIFS = True

T_LOW, T_HIGH = 77, 160
P_LOW, P_HIGH = 2, 100
FORCEFIELD = None

MAX_MISSING = 0.4
ROBUST_Z = 8

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PERM_ROWS = 3000
TOP_FEATURES = 6
MIN_ROWS = 100
LEARNING_CURVE_MAX_ROWS = 20000
TOP_N = 10
SHOW_PLOTS = True

DATA = Path("mofxdb_data")
CIF_DIR = DATA / "cifs"
OUT = Path("results")
MODEL_DIR = Path("models")

NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te", "I", "Xe"}
FRAC_ELEMENTS = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]
ID_COLS = {"name", "mofid", "mofkey", "database", "metals"}
TARGET_PREFIX = ("uptake", "cap_", "del_")
VALID = {
    "void_fraction": (1e-6, 1),
    "density_gcm3": (0.05, 5),
    "surface_area_m2g": (0, np.inf),
    "surface_area_m2cm3": (0, np.inf),
    "pld": (1e-6, np.inf),
    "lcd": (1e-6, np.inf),
    "volume_A3": (1, np.inf),
}
OUTLIER_COLS = ["density_gcm3", "volume_per_atom_A3", "pore_volume_cm3g", "metal_density_nm3", "sa_per_pore_volume"]
TASKS = {
    "cap_gL": "ml_wide_clean.csv",
    "cap_wt": "ml_wide_clean.csv",
    "del_ps_gL": "ml_wide_clean.csv",
    "del_ps_wt": "ml_wide_clean.csv",
    "del_tps_gL": "ml_wide_clean.csv",
    "del_tps_wt": "ml_wide_clean.csv",
    "uptake_gL": "ml_long_clean.csv",
    "uptake_mgg": "ml_long_clean.csv",
}


def feature_columns(df, varying=True):
    return [c for c in df.select_dtypes("number").columns
            if c not in ID_COLS and not c.startswith(TARGET_PREFIX) and (not varying or df[c].nunique() > 1)]


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def fetch_data():
    section(f"1. FETCH {LIMIT} {DATABASE} MOFs FROM MOFX-DB")
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    rows, points = [], []
    try:
        for i, mof in enumerate(fetch(database=DATABASE, limit=LIMIT, loading_unit=LOADING_UNIT,
                                      pressure_unit=PRESSURE_UNIT, telemetry=False), 1):
            elements = sorted({e.symbol for e in mof.elements})
            rows.append({
                "name": mof.name, "mofid": mof.mofid, "mofkey": mof.mofkey, "database": mof.database,
                "void_fraction": mof.void_fraction, "surface_area_m2g": mof.surface_area_m2g,
                "surface_area_m2cm3": mof.surface_area_m2cm3, "pld": mof.pld, "lcd": mof.lcd,
                "n_elements": len(elements), "metals": "|".join(e for e in elements if e not in NONMETALS),
                **{f"el_{e}": 1 for e in elements},
            })
            for iso in mof.isotherms:
                if [a.formula for a in iso.adsorbates] != [GAS]:
                    continue
                for p in iso.isotherm_data:
                    points.append({"name": mof.name, "temperature_K": iso.temperature, "pressure": p.pressure,
                                   "uptake": p.total_adsorption, "pressure_unit": iso.pressureUnits,
                                   "uptake_unit": iso.adsorptionUnits,
                                   "molecule_forcefield": iso.molecule_forcefield,
                                   "adsorbent_forcefield": iso.adsorbent_forcefield})
            if SAVE_CIFS:
                (CIF_DIR / f"{mof.name}.cif").write_text(mof.cif)
            if i % 500 == 0:
                print(f"{i}/{LIMIT}")
    finally:
        if rows:
            df = pd.DataFrame(rows)
            el = [c for c in df.columns if c.startswith("el_")]
            df[el] = df[el].fillna(0).astype(int)
            df.to_csv(DATA / "mofs_features_targets.csv", index=False)
            pd.DataFrame(points).to_csv(DATA / f"{GAS}_isotherms_long.csv", index=False)
            print(f"saved {len(df)} MOFs, {len(points)} {GAS} points")


def cif_features(path):
    try:
        s = Structure.from_file(path)
    except Exception:
        return None
    comp = s.composition
    n = comp.num_atoms
    n_metal = sum(amt for el, amt in comp.items() if el.is_metal)
    lat = s.lattice
    return {
        "name": path.stem, "density_gcm3": s.density, "volume_A3": s.volume, "n_atoms": n,
        "mass_amu": comp.weight, "volume_per_atom_A3": s.volume / n,
        "a_A": lat.a, "b_A": lat.b, "c_A": lat.c, "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma,
        "metal_fraction": n_metal / n, "metal_density_nm3": n_metal / (s.volume / 1000),
        "avg_electronegativity": comp.average_electroneg, "avg_atomic_mass": comp.weight / n,
        **{f"frac_{e}": comp.get_atomic_fraction(e) for e in FRAC_ELEMENTS},
    }


def uptake_at(lookup, name, t, p):
    curve = lookup.get((name, t))
    if curve is None or not curve[2] <= p <= curve[3]:
        return np.nan
    return float(np.interp(np.log(p), curve[0], curve[1]))


def add_units(df, col):
    df[f"{col}_gL"] = df[f"{col}_mgg"] * df["density_gcm3"]
    df[f"{col}_wt"] = 100 * df[f"{col}_mgg"] / (1000 + df[f"{col}_mgg"])


def build_features():
    section("2. BUILD FEATURES AND TARGETS")
    base = pd.read_csv(DATA / "mofs_features_targets.csv")
    base = base.drop(columns=[c for c in base.columns if c.startswith(f"{GAS}_")])
    with ProcessPoolExecutor() as ex:
        feats = [r for r in ex.map(cif_features, sorted(CIF_DIR.glob("*.cif")), chunksize=64) if r]
    struct = base.merge(pd.DataFrame(feats), on="name", how="inner")
    struct["pore_volume_cm3g"] = struct["void_fraction"] / struct["density_gcm3"]
    struct["lcd_pld_ratio"] = struct["lcd"] / struct["pld"]
    struct["sa_per_pore_volume"] = struct["surface_area_m2g"] / struct["pore_volume_cm3g"]
    struct = struct.replace([np.inf, -np.inf], np.nan)

    iso = pd.read_csv(DATA / f"{GAS}_isotherms_long.csv")
    if iso["molecule_forcefield"].notna().any():
        ff = FORCEFIELD or iso["molecule_forcefield"].mode().iat[0]
        iso = iso[iso["molecule_forcefield"] == ff]
        print(f"forcefield: {ff}")
    iso = (iso.groupby(["name", "temperature_K", "pressure"], as_index=False)["uptake"].mean()
              .rename(columns={"uptake": "uptake_mgg", "pressure": "pressure_bar"}))
    print("H2 conditions (K, bar):", sorted({(t, p) for t, p in zip(iso["temperature_K"], iso["pressure_bar"])}))

    long = iso.merge(struct, on="name", how="inner")
    long["inv_temperature"] = 1 / long["temperature_K"]
    long["log_pressure"] = np.log(long["pressure_bar"])
    add_units(long, "uptake")
    long.to_csv(DATA / "ml_long.csv", index=False)

    lookup = {k: (np.log(g["pressure_bar"].to_numpy()), g["uptake_mgg"].to_numpy(),
                  g["pressure_bar"].min(), g["pressure_bar"].max())
              for k, g in iso.sort_values("pressure_bar").groupby(["name", "temperature_K"])}
    tgt = []
    for name in struct["name"]:
        hi = uptake_at(lookup, name, T_LOW, P_HIGH)
        tgt.append({"name": name, "cap_mgg": hi,
                    "del_ps_mgg": hi - uptake_at(lookup, name, T_LOW, P_LOW),
                    "del_tps_mgg": hi - uptake_at(lookup, name, T_HIGH, P_LOW)})
    wide = struct.merge(pd.DataFrame(tgt), on="name")
    for col in ["cap", "del_ps", "del_tps"]:
        add_units(wide, col)
    wide.to_csv(DATA / "ml_wide.csv", index=False)
    print(f"long table: {long.shape}  wide table: {wide.shape}")
    print(wide.filter(regex="^(cap|del)_").notna().sum().to_string())


def clean_data():
    section("3. CLEAN DATA")
    wide = pd.read_csv(DATA / "ml_wide.csv")
    long = pd.read_csv(DATA / "ml_long.csv")

    miss = wide[feature_columns(wide, varying=False)].isna().mean().sort_values()
    miss = miss[miss > 0]
    if not miss.empty:
        plt.figure(figsize=(7, max(3, 0.3 * len(miss))))
        plt.barh(miss.index, miss.values * 100)
        plt.axvline(MAX_MISSING * 100, color="red", ls="--", label="drop threshold")
        plt.xlabel("Missing (%)")
        plt.legend()
        save("missing_values")

    report = []

    def log(step, before, df):
        report.append({"step": step, "removed": before - len(df), "remaining": len(df)})
        return len(df)

    n = log("start", len(wide), wide)
    if DEDUP_BY:
        wide = wide[~(wide[DEDUP_BY].notna() & wide.duplicated(DEDUP_BY))]
        n = log(f"duplicate {DEDUP_BY}", n, wide)
    for col, (lo, hi) in VALID.items():
        if col in wide:
            wide = wide[wide[col].isna() | wide[col].between(lo, hi)]
            n = log(f"invalid {col}", n, wide)
    wide = wide[wide["lcd"].isna() | wide["pld"].isna() | (wide["lcd"] >= wide["pld"] - 0.01)]
    n = log("lcd < pld", n, wide)
    for col in OUTLIER_COLS:
        if col in wide:
            x = wide[col]
            mad = (x - x.median()).abs().median()
            if mad > 0:
                wide = wide[x.isna() | (0.6745 * (x - x.median()).abs() / mad <= ROBUST_Z)]
                n = log(f"outlier {col}", n, wide)

    sparse = [c for c in feature_columns(wide, varying=False) if wide[c].isna().mean() > MAX_MISSING]
    wide = wide.drop(columns=sparse)
    report.append({"step": f"dropped sparse features: {sparse}", "removed": 0, "remaining": len(wide)})
    targets = [c for c in wide.columns if c.startswith(("cap_", "del_"))]
    wide[targets] = wide[targets].mask(wide[targets] < 0)

    n = len(long)
    long = long[long["name"].isin(wide["name"])].drop(columns=[c for c in sparse if c in long])
    n = log("long table: MOFs removed above", n, long)
    long = long[long["uptake_mgg"] >= 0]
    log("long table: negative uptake", n, long)

    rep = pd.DataFrame(report)
    rep.to_csv(OUT / "cleaning_report.csv", index=False)
    display(rep)

    cols = [c for c in wide.columns if c.endswith("_gL") and wide[c].notna().any()]
    data = [(c, wide[c]) for c in cols] + [("uptake_gL (all points)", long["uptake_gL"])]
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 3.2))
    for ax, (name, s) in zip(np.atleast_1d(axes), data):
        ax.hist(s.dropna(), bins=50)
        ax.set_title(name)
        ax.set_xlabel("g/L")
    save("target_distributions")

    wide.to_csv(DATA / "ml_wide_clean.csv", index=False)
    long.to_csv(DATA / "ml_long_clean.csv", index=False)


def make_models():
    models = {
        "linear": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                               RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                                     max_features=0.5, n_jobs=-1,
                                                                     random_state=SEED)),
        "hist_gb": lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED),
    }
    if XGBRegressor is not None:
        models["xgboost"] = lambda: XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                                                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                                 random_state=SEED)
    return models


MODELS = make_models()


def plot_parity_grid(y, preds, scores, target):
    lo = min(y.min(), min(p.min() for p in preds.values()))
    hi = max(y.max(), max(p.max() for p in preds.values()))
    fig, axes = plt.subplots(1, len(preds), figsize=(3.6 * len(preds), 3.8), sharex=True, sharey=True)
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(y, p, s=4, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{name}\nR² = {scores[name]:.3f}")
        ax.set_xlabel(f"True {target}")
    np.atleast_1d(axes)[0].set_ylabel(f"Predicted {target}")
    save(f"parity_{target}")


def plot_residuals(y, pred, target, model_name):
    res = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, res, s=4, alpha=0.3)
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel(f"True {target}")
    axes[0].set_ylabel("Predicted − true")
    axes[0].set_title(f"Residuals · {model_name}")
    axes[1].hist(res, bins=60)
    axes[1].set_xlabel("Predicted − true")
    axes[1].set_title(f"Error distribution · MAE = {np.abs(res).mean():.2f}")
    save(f"residuals_{target}")


def plot_learning_curve(make, X, y, groups, target, model_name):
    sizes, tr_s, va_s = learning_curve(make(), X, y, groups=groups, cv=GroupKFold(3),
                                       train_sizes=np.linspace(0.1, 1.0, 6), scoring="r2")
    plt.figure(figsize=(6, 4))
    for s, label in [(tr_s, "train"), (va_s, "validation")]:
        m, sd = s.mean(axis=1), s.std(axis=1)
        plt.plot(sizes, m, "o-", label=label)
        plt.fill_between(sizes, m - sd, m + sd, alpha=0.2)
    plt.xlabel("Training rows")
    plt.ylabel("R²")
    plt.title(f"Learning curve · {target} · {model_name}")
    plt.legend()
    save(f"learning_curve_{target}")


def plot_importance(model, X, y, target, model_name):
    Xs = X.sample(min(PERM_ROWS, len(X)), random_state=SEED)
    r = permutation_importance(model, Xs, y.loc[Xs.index], n_repeats=5, scoring="r2", random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
             .sort_values("importance", ascending=False))
    imp.to_csv(OUT / f"importance_{target}.csv", index=False)
    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top["feature"], top["importance"], xerr=top["std"])
    plt.xlabel("Permutation importance (drop in R²)")
    plt.title(f"{target} · {model_name}")
    save(f"importance_{target}")
    return imp


def plot_features(df, target, features):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, f in zip(axes.ravel(), features):
        ax.scatter(df[f], df[target], s=4, alpha=0.3)
        ax.set_xlabel(f)
        ax.set_ylabel(target)
        ax.set_title(f"Spearman ρ = {df[[f, target]].corr(method='spearman').iat[0, 1]:.2f}")
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle(f"Top features vs {target}")
    save(f"features_{target}")


def plot_model_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col, label in [(axes[0], "test_r2", "Test R²"), (axes[1], "cv_r2_mean", "5-fold CV R² (train set)")]:
        pivot = results.pivot(index="target", columns="model", values=col)
        pivot.plot(kind="bar", ax=ax, width=0.8, legend=False)
        ax.set_ylabel(label)
        ax.set_ylim(min(0, pivot.min().min()), 1)
        ax.axhline(0, color="black", lw=0.8)
        ax.tick_params(axis="x", rotation=20)
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1, 1))
    save("model_comparison")


def plot_correlation():
    df = pd.read_csv(DATA / "ml_wide_clean.csv")
    targets = [c for c in df.columns if c.endswith(("_gL", "_wt")) and df[c].notna().sum() > 10]
    if not targets:
        return
    feats = feature_columns(df)
    corr = df[feats + targets].corr(method="spearman").loc[feats, targets]
    corr = corr.reindex(corr.abs().max(axis=1).sort_values().index)
    plt.figure(figsize=(1.2 * len(targets) + 4, 0.3 * len(feats) + 2))
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Spearman ρ")
    plt.xticks(range(len(targets)), targets, rotation=30, ha="right")
    plt.yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(targets)):
            plt.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    save("feature_target_correlation")


def train_target(target, file):
    df = pd.read_csv(DATA / file)
    if target not in df or df[target].notna().sum() < MIN_ROWS:
        print(f"skip {target}: fewer than {MIN_ROWS} rows")
        return [], None
    section(f"TARGET: {target}")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    cols = feature_columns(df)
    X, y = df[cols], df[target]
    groups = df["mofkey"].fillna(df["name"])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(X, y, groups))
    Xtr, ytr, Xte, yte, gtr = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], groups.iloc[tr]

    rows, fitted = [], {}
    for name, make in MODELS.items():
        model = make().fit(Xtr, ytr)
        p_tr, p_te = model.predict(Xtr), model.predict(Xte)
        cv = cross_val_score(make(), Xtr, ytr, groups=gtr, cv=GroupKFold(CV_FOLDS), scoring="r2")
        rows.append({"target": target, "model": name, "n_features": len(cols), "n_train": len(tr), "n_test": len(te),
                     "train_r2": r2_score(ytr, p_tr), "test_r2": r2_score(yte, p_te),
                     "test_mae": mean_absolute_error(yte, p_te), "test_rmse": root_mean_squared_error(yte, p_te),
                     "cv_r2_mean": cv.mean(), "cv_r2_std": cv.std()})
        fitted[name] = (model, p_te)
        print(f"{name:14s} train R² {rows[-1]['train_r2']:.3f}  test R² {rows[-1]['test_r2']:.3f}  "
              f"MAE {rows[-1]['test_mae']:.3f}  CV R² {cv.mean():.3f} ± {cv.std():.3f}")

    best = max(rows, key=lambda r: r["cv_r2_mean"])
    best_model, best_pred = fitted[best["model"]]
    print(f"best: {best['model']}")

    plot_parity_grid(yte, {k: v[1] for k, v in fitted.items()}, {r["model"]: r["test_r2"] for r in rows}, target)
    plot_residuals(yte.to_numpy(), best_pred, target, best["model"])
    if len(tr) <= LEARNING_CURVE_MAX_ROWS:
        plot_learning_curve(MODELS[best["model"]], Xtr, ytr, gtr, target, best["model"])
    imp = plot_importance(best_model, Xte, yte, target, best["model"])
    plot_features(df, target, imp["feature"].head(TOP_FEATURES).tolist())

    pred = df.iloc[te][["name"] + [c for c in ("temperature_K", "pressure_bar") if c in df]].copy()
    pred["true"], pred["predicted"] = yte.to_numpy(), best_pred
    pred["error"] = pred["predicted"] - pred["true"]
    pred.to_csv(OUT / f"predictions_{target}.csv", index=False)
    joblib.dump({"model": best_model, "model_name": best["model"], "features": cols, "target": target,
                 "metrics": best}, MODEL_DIR / f"{target}.joblib")
    return rows, best


def train_all():
    section("4. TRAIN AND EVALUATE MODELS")
    rows, best = [], []
    for t, f in TASKS.items():
        r, b = train_target(t, f)
        rows += r
        if b:
            best.append(b)
    results = pd.DataFrame(rows)
    best = pd.DataFrame(best)
    results.to_csv(OUT / "model_results.csv", index=False)
    best.to_csv(OUT / "best_models.csv", index=False)
    cols = ["target", "model", "n_train", "n_test", "train_r2", "test_r2", "test_mae", "test_rmse",
            "cv_r2_mean", "cv_r2_std"]
    section("MODEL RESULTS — ALL MODELS")
    display(results[cols].round(3))
    section("MODEL RESULTS — BEST MODEL PER TARGET (chosen by CV R²)")
    display(best[cols].round(3))
    plot_model_comparison(results)
    plot_correlation()
    return best


def predict(df, target):
    bundle = joblib.load(MODEL_DIR / f"{target}.joblib")
    return bundle["model"].predict(df.reindex(columns=bundle["features"]))


def screen(targets):
    section("5. PREDICT AND RANK MOFs WITH SAVED MODELS")
    df = pd.read_csv(DATA / "ml_wide_clean.csv")
    for t in targets:
        df[f"pred_{t}"] = predict(df, t)
        keep = ["name", f"pred_{t}", t, "void_fraction", "density_gcm3", "surface_area_m2g", "lcd"]
        top = df.sort_values(f"pred_{t}", ascending=False)[[c for c in keep if c in df]].head(TOP_N)
        print(f"\nTop {TOP_N} MOFs by predicted {t}")
        display(top.round(3))
    df[["name"] + [f"pred_{t}" for t in targets]].to_csv(OUT / "screening_predictions.csv", index=False)


def critical_checks():
    section("6. CRITICAL CHECKS: DESCRIPTOR SETS AND EXTRAPOLATION")
    if XGBRegressor is None:
        print("xgboost not installed, skipping")
        return
    df_all = pd.read_csv(DATA / "ml_wide_clean.csv", low_memory=False)
    full = feature_columns(df_all)
    geom = ["void_fraction", "surface_area_m2g", "surface_area_m2cm3", "pld", "lcd", "density_gcm3", "pore_volume_cm3g"]
    sets = {"Void fraction only": ["void_fraction"], "7 geometric features": geom, f"All {len(full)} features": full}
    rows, ext = [], []
    for target in ["cap_gL", "del_ps_gL", "cap_wt", "del_ps_wt"]:
        df = df_all.dropna(subset=[target]).reset_index(drop=True)
        y, groups = df[target], df["mofkey"].fillna(df["name"])
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(df, y, groups))
        for name, cols in sets.items():
            p = MODELS["xgboost"]().fit(df.iloc[tr][cols], y.iloc[tr]).predict(df.iloc[te][cols])
            rows.append({"target": target, "features": name, "n_features": len(cols),
                         "test_r2": r2_score(y.iloc[te], p), "test_mae": mean_absolute_error(y.iloc[te], p)})
        cut = y.quantile(0.9)
        top = y >= cut
        p_rand = MODELS["xgboost"]().fit(df.iloc[tr][full], y.iloc[tr]).predict(df.iloc[te][full])
        mask = top.iloc[te].to_numpy()
        p_ext = MODELS["xgboost"]().fit(df.loc[~top, full], y[~top]).predict(df.loc[top, full])
        ext.append({"target": target, "top10_threshold": cut, "n_top": int(top.sum()),
                    "mae_top10_random_split": mean_absolute_error(y.iloc[te][mask], p_rand[mask]),
                    "mae_top10_extrapolation": mean_absolute_error(y[top], p_ext),
                    "max_true_top": y[top].max(), "max_pred_extrapolation": p_ext.max()})
        if target == "del_ps_gL":
            plt.figure(figsize=(5.2, 4.6))
            plt.scatter(y[top], p_ext, s=8, alpha=0.5)
            lo, hi = min(y[top].min(), p_ext.min()), y[top].max()
            plt.plot([lo, hi], [lo, hi], "r--", lw=1)
            plt.axhline(cut, color="grey", ls=":", lw=1)
            plt.xlabel("True deliverable capacity (g/L)")
            plt.ylabel("Predicted (trained on bottom 90%)")
            plt.title(f"Extrapolation to top-10% MOFs\nMAE = {ext[-1]['mae_top10_extrapolation']:.2f} g/L")
            save("extrapolation_del_ps_gL")
    base, ext = pd.DataFrame(rows), pd.DataFrame(ext)
    base.to_csv(OUT / "feature_set_baseline.csv", index=False)
    ext.to_csv(OUT / "extrapolation_test.csv", index=False)
    display(base.round(4))
    display(ext.round(3))


def main():
    for d in (DATA, OUT, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if REFETCH or not (DATA / "mofs_features_targets.csv").exists():
        fetch_data()
    else:
        print("using existing data in mofxdb_data/ (set REFETCH = True to download again)")
    build_features()
    clean_data()
    best = train_all()
    screen([t for t in best["target"] if t in ("cap_gL", "del_ps_gL", "del_tps_gL")])
    critical_checks()


if __name__ == "__main__":
    main()
