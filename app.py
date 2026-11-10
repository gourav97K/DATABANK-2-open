# ==============================================================================
# SAVE THIS FILE AS: app.py
# RUN IN TERMINAL  : streamlit run app.py
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import streamlit as st

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="NVH Acoustic Material Predictor & Data Bank", layout="wide")

# ==============================================================================
# CONSTANTS & CORE ACOUSTIC HELPERS
# ==============================================================================
RAW_FILE = "sample_data_bank.xlsx"
DIR_FILE = "directional_data_bank.xlsx"
FREQ     = [100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000]
STC_FREQ = [125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000]
STC_OFF  = np.array([-16,-13,-10,-7,-4,-1,0,1,2,3,4,4,4,4,4,4])

LAYER_TYPES = [
    "PU Foam", "Melamine Foam", "Glass Wool", "PET Felt", "Shoddy Cotton",
    "MLV Barrier", "EPDM Barrier", "Nonwoven Scrim", "Aluminium Foil"
]

TYPICAL = {   # role, flow resistivity, porosity, tortuosity, visc. length um, therm. length um, density kg/m3
    "PU Foam":        ("absorber", 12000, 0.96, 1.2,  90, 220,   40),
    "Melamine Foam":  ("absorber", 12000, 0.98, 1.0,  90, 200,   10),
    "Glass Wool":     ("absorber", 20000, 0.99, 1.0,  75, 180,   30),
    "PET Felt":       ("absorber", 30000, 0.94, 1.5,  65, 150,  120),
    "Shoddy Cotton":  ("absorber", 40000, 0.92, 1.6,  55, 130,  150),
    "MLV Barrier":    ("barrier",    1e9, 0.01, 1.0,   1,   1, 2000),
    "EPDM Barrier":   ("barrier",    1e9, 0.01, 1.0,   1,   1, 1500),
    "Nonwoven Scrim": ("facing",     600, 0.90, 1.1, 110, 300,   60),
    "Aluminium Foil": ("facing",     1e9, 0.01, 1.0,   1,   1, 2700),
}

def format_freq_axis(a):
    a.set_xscale("log")
    a.set_xticks(FREQ)
    a.get_xaxis().set_major_formatter(ScalarFormatter())
    a.get_xaxis().set_minor_formatter(NullFormatter())
    a.set_xticklabels([str(f) for f in FREQ], rotation=45, fontsize=8)
    a.set_xlim(90, 5500)
    a.set_xlabel("Frequency (Hz)")
    a.grid(True, which="both", alpha=0.3)

def calc_nrc(sac_18):
    idx = [FREQ.index(f) for f in (250, 500, 1000, 2000)]
    return round(float(np.round(np.mean(np.asarray(sac_18)[idx]) / 0.05) * 0.05), 2)

def calc_stc(tl_18):
    idx = [FREQ.index(f) for f in STC_FREQ]
    tl_16 = np.asarray(tl_18, dtype=float)[idx]
    for stc in range(100, -10, -1):
        defic = np.maximum(0.0, (stc + STC_OFF) - tl_16)
        if defic.sum() <= 32.0 and defic.max() <= 8.0:
            return int(stc)
    return 0

def _delany_bazley_sac_prior(d_m, sigma, f_hz, has_barrier=False, has_facing=False, facing_perf_pct=0.0):
    if d_m <= 1e-5 or sigma <= 0:
        return np.full_like(f_hz, 0.01, dtype=float)
    c0, rho0 = 343.0, 1.21
    X = np.clip(rho0 * f_hz / sigma, 1e-4, 10.0)
    Zc = 1.0 + 0.0571 * (X ** -0.754) - 1j * 0.087 * (X ** -0.732)
    gamma = (2.0 * np.pi * f_hz / c0) * (0.189 * (X ** -0.595) + 1j * (1.0 + 0.0978 * (X ** -0.700)))
    Zs = Zc / np.tanh(gamma * d_m)
    R = (Zs - 1.0) / (Zs + 1.0)
    alpha = 1.0 - np.abs(R) ** 2
    if has_facing and facing_perf_pct <= 0.1:
        alpha = alpha * 0.80
    if has_barrier:
        alpha = alpha * 0.55
    return np.clip(alpha, 0.01, 0.99)

def featurize(layers):
    L_rcv_to_src = pd.DataFrame(layers).reset_index(drop=True)
    L_src_to_rcv = L_rcv_to_src.iloc[::-1].reset_index(drop=True)

    tot_t = float(L_src_to_rcv.thickness_mm.sum())
    tot_g = float(L_src_to_rcv.gsm.sum())
    f = {
        "total_thickness": tot_t,
        "total_gsm": tot_g,
        "total_density": tot_g / max(tot_t, 1e-6),
        "n_layers": len(L_src_to_rcv),
        "log_total_gsm": np.log10(tot_g + 1.0),
        "log_total_thickness": np.log10(tot_t + 1.0),
    }
    for t in LAYER_TYPES:
        s = L_src_to_rcv[L_src_to_rcv.layer_material == t]
        f[f"t_{t}"], f[f"gsm_{t}"] = float(s.thickness_mm.sum()), float(s.gsm.sum())

    barrier_rows = L_src_to_rcv.index[L_src_to_rcv.role == "barrier"].tolist()
    has_barrier = int(len(barrier_rows) > 0)
    f["has_barrier"] = has_barrier

    if has_barrier:
        b_idx = barrier_rows[0]
        src_layers = L_src_to_rcv.iloc[:b_idx]
        rcv_layers = L_src_to_rcv.iloc[b_idx + 1:]
    else:
        src_layers = L_src_to_rcv
        rcv_layers = L_src_to_rcv.iloc[0:0]

    src_ab = src_layers[src_layers.role == "absorber"]
    src_ab_t = float(src_ab.thickness_mm.sum())
    f["src_ab_thickness"] = src_ab_t
    f["src_ab_gsm"] = float(src_ab.gsm.sum())
    if len(src_ab) > 0 and src_ab_t > 0:
        wt_src = src_ab.thickness_mm / src_ab_t
        for p in ["flow_resistivity", "porosity", "tortuosity", "visc_char_len_um", "therm_char_len_um", "density_kg_m3"]:
            f[f"src_ab_{p}"] = float((src_ab[p] * wt_src).sum())
    else:
        for p in ["flow_resistivity", "porosity", "tortuosity", "visc_char_len_um", "therm_char_len_um", "density_kg_m3"]:
            f[f"src_ab_{p}"] = 0.0
    f["src_ab_log_sigma"] = np.log10(f["src_ab_flow_resistivity"] + 1.0)

    rcv_ab = rcv_layers[rcv_layers.role == "absorber"]
    rcv_ab_t = float(rcv_ab.thickness_mm.sum())
    f["rcv_ab_thickness"] = rcv_ab_t
    f["rcv_ab_gsm"] = float(rcv_ab.gsm.sum())
    if len(rcv_ab) > 0 and rcv_ab_t > 0:
        wt_rcv = rcv_ab.thickness_mm / rcv_ab_t
        for p in ["flow_resistivity", "porosity", "tortuosity", "visc_char_len_um", "therm_char_len_um", "density_kg_m3"]:
            f[f"rcv_ab_{p}"] = float((rcv_ab[p] * wt_rcv).sum())
    else:
        for p in ["flow_resistivity", "porosity", "tortuosity", "visc_char_len_um", "therm_char_len_um", "density_kg_m3"]:
            f[f"rcv_ab_{p}"] = 0.0

    all_ab = L_src_to_rcv[L_src_to_rcv.role == "absorber"]
    all_ab_t = float(all_ab.thickness_mm.sum())
    f["ab_thickness"] = all_ab_t
    if len(all_ab) > 0 and all_ab_t > 0:
        wt_all = all_ab.thickness_mm / all_ab_t
        for p in ["flow_resistivity", "porosity", "tortuosity", "visc_char_len_um", "therm_char_len_um", "density_kg_m3"]:
            f[f"ab_{p}"] = float((all_ab[p] * wt_all).sum())
    else:
        for p in ["flow_resistivity", "porosity", "tortuosity", "visc_char_len_um", "therm_char_len_um", "density_kg_m3"]:
            f[f"ab_{p}"] = 0.0
    f["ab_log_sigma"] = np.log10(f["ab_flow_resistivity"] + 1.0)

    has_facing = int((L_src_to_rcv.role == "facing").any())
    fac_layers = L_src_to_rcv[L_src_to_rcv.role == "facing"]
    fac_perf_pct = float(fac_layers.perforation_pct.max()) if has_facing else 0.0
    f["has_facing"]           = has_facing
    f["fac_perf_pct"]         = fac_perf_pct
    f["src_first_is_facing"]  = int(L_src_to_rcv.iloc[0].role == "facing")
    f["src_first_is_barrier"] = int(L_src_to_rcv.iloc[0].role == "barrier")
    f["barrier_gsm"]          = float(L_src_to_rcv[L_src_to_rcv.role == "barrier"].gsm.sum())
    f["is_sandwich"]          = int(has_barrier and src_ab_t > 0 and rcv_ab_t > 0)
    f["perf_pct_max"]         = float(L_src_to_rcv.perforation_pct.max())
    f["perf_dia_max"]         = float(L_src_to_rcv.perforation_dia_mm.max())

    f_arr = np.array(FREQ, dtype=float)
    sac_prior_all = _delany_bazley_sac_prior(
        d_m=all_ab_t * 1e-3, sigma=max(f["ab_flow_resistivity"], 100.0),
        f_hz=f_arr, has_barrier=bool(has_barrier), has_facing=bool(has_facing), facing_perf_pct=fac_perf_pct
    )
    sac_prior_src = _delany_bazley_sac_prior(
        d_m=src_ab_t * 1e-3, sigma=max(f["src_ab_flow_resistivity"], 100.0),
        f_hz=f_arr, has_barrier=bool(has_barrier),
        has_facing=bool((src_layers.role == "facing").any()),
        facing_perf_pct=float(src_layers[src_layers.role == "facing"].perforation_pct.max()) if (src_layers.role == "facing").any() else 0.0
    )
    surface_mass_kg_m2 = max(tot_g * 1e-3, 0.05)
    tl_mass_law = np.maximum(0.0, 20.0 * np.log10(f_arr * surface_mass_kg_m2) - 42.4 + 6.0 * f["is_sandwich"])

    for idx_f, freq_val in enumerate(FREQ):
        f[f"prior_sac_all_{freq_val}"] = float(sac_prior_all[idx_f])
        f[f"prior_sac_src_{freq_val}"] = float(sac_prior_src[idx_f])
        f[f"prior_tl_{freq_val}"]      = float(tl_mass_law[idx_f])

    return f

def _make_scaled_mlp(hidden_layer_sizes, activation, alpha, random_state):
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver="lbfgs", alpha=alpha, max_iter=1500, random_state=random_state)
        ),
        transformer=StandardScaler()
    )

class AcousticHybridModel(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, Y):
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)
        self.estimators_dict_ = {
            "MLP-ReLU (128, 64) Neural Net": _make_scaled_mlp((128, 64), "relu", 8.0, self.random_state),
            "MLP-Tanh (96, 48) Neural Net":  _make_scaled_mlp((96, 48),  "tanh", 8.0, self.random_state + 7),
            "ExtraTrees (300 trees)":        ExtraTreesRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
            "RandomForest (300 trees)":      RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
        }
        self.weights_ = [0.30, 0.30, 0.25, 0.15]
        for m in self.estimators_dict_.values():
            m.fit(X_arr, Y_arr)
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        preds = [w * m.predict(X_arr) for w, m in zip(self.weights_, self.estimators_dict_.values())]
        out = np.sum(preds, axis=0)
        out[:, :18] = np.clip(out[:, :18], 0.01, 0.99)
        return out

    def predict_std(self, X):
        X_arr = np.asarray(X, dtype=float)
        preds = []
        for m in self.estimators_dict_.values():
            p = m.predict(X_arr).copy()
            p[:, :18] = np.clip(p[:, :18], 0.01, 0.99)
            preds.append(p)
        return np.stack(preds, axis=0).std(axis=0)

# ==============================================================================
# CACHED DATA LOADER & MODEL TRAINER
# ==============================================================================
@st.cache_resource(show_spinner="Loading acoustic databank and training Hybrid Neural-Tree Model...")
def load_and_train(excel_source):
    d = pd.read_excel(excel_source, sheet_name=None)
    mat, lay = d["Materials"].copy(), d["Layers"].copy()
    sac, tl = d["SAC_curves"].copy(), d["TL_IL_curves"].copy()

    # Convert to directional [RCV] Left <-- Right [SRC] if not already converted
    if "source_side_material" not in mat.columns:
        new_layers, src_mats, rcv_mats, dir_configs = [], [], [], []
        for mid in mat["material_id"]:
            sub = lay[lay["material_id"] == mid].sort_values("layer_no").iloc[::-1].reset_index(drop=True)
            n = len(sub)
            sub["layer_no"] = np.arange(1, n + 1)
            barrier_idx = sub.index[sub["role"] == "barrier"].tolist()
            sides = []
            for idx_i in range(n):
                if barrier_idx:
                    b_pos = barrier_idx[0]
                    sides.append("source_side" if idx_i > b_pos else ("barrier_core" if idx_i == b_pos else "receiver_side"))
                else:
                    sides.append("source_side" if idx_i == n - 1 else ("receiver_side" if idx_i == 0 else "core_absorber"))
            sub["acoustic_side"] = sides
            new_layers.append(sub)
            rcv_mats.append(f"{sub.iloc[0]['layer_material']} ({sub.iloc[0]['thickness_mm']}mm)")
            src_mats.append(f"{sub.iloc[-1]['layer_material']} ({sub.iloc[-1]['thickness_mm']}mm)")
            dir_configs.append("[RCV] " + " | ".join(f"{r.layer_material} {r.thickness_mm}mm" for _, r in sub.iterrows()) + " [SRC]")
        lay = pd.concat(new_layers, ignore_index=True)
        mat["receiver_side_material"] = rcv_mats
        mat["source_side_material"]   = src_mats
        mat["layer_config"]           = dir_configs

    X = pd.DataFrame([
        featurize(lay[lay.material_id == m].sort_values("layer_no").to_dict("records"))
        for m in mat.material_id
    ])
    Y = pd.concat([
        sac.set_index("material_id").loc[mat.material_id].reset_index(drop=True),
        tl.set_index("material_id").loc[mat.material_id].reset_index(drop=True)
    ], axis=1)

    groups = (mat.customer_name + "|" + mat.material_name).values
    c_sac  = [k for k in Y.columns if k.startswith("SAC_")]
    c_tl   = [k for k in Y.columns if k.startswith("TL_")]
    c_il   = [k for k in Y.columns if k.startswith("IL_")]

    candidate_models = {
        "1. MLP-ReLU (128, 64) Neural Net":      _make_scaled_mlp((128, 64), "relu", 8.0, 0),
        "2. MLP-Tanh (96, 48) Neural Net":       _make_scaled_mlp((96, 48),  "tanh", 8.0, 7),
        "3. ExtraTrees (300 trees)":             ExtraTreesRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        "4. RandomForest (300 trees)":           RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        "5. AcousticHybridModel (USED IN PROD)": AcousticHybridModel(random_state=0),
    }

    rows_bench = []
    for m_name, m_obj in candidate_models.items():
        p_cv = pd.DataFrame(cross_val_predict(m_obj, X, Y, cv=GroupKFold(5), groups=groups), columns=Y.columns)
        p_cv[c_sac] = np.clip(p_cv[c_sac], 0.01, 0.99)
        cv_nrc = [calc_nrc(p_cv.iloc[i][c_sac].values) for i in range(len(p_cv))]
        cv_stc = [calc_stc(p_cv.iloc[i][c_tl].values)  for i in range(len(p_cv))]
        rows_bench.append({
            "Model": m_name,
            "SAC R2": round(r2_score(Y[c_sac], p_cv[c_sac], multioutput="variance_weighted"), 3),
            "SAC MAE": round(mean_absolute_error(Y[c_sac], p_cv[c_sac]), 4),
            "TL R2": round(r2_score(Y[c_tl], p_cv[c_tl]), 3),
            "TL MAE (dB)": round(mean_absolute_error(Y[c_tl], p_cv[c_tl]), 2),
            "IL R2": round(r2_score(Y[c_il], p_cv[c_il]), 3),
            "IL MAE (dB)": round(mean_absolute_error(Y[c_il], p_cv[c_il]), 2),
            "NRC R2": round(r2_score(mat.nrc, cv_nrc), 3),
            "STC R2": round(r2_score(mat.stc, cv_stc), 3),
        })

    model = AcousticHybridModel(random_state=0)
    model.fit(X, Y)
    return mat, lay, sac, tl, X, Y, model, pd.DataFrame(rows_bench)

# Sidebar File Uploader or Default Local File
st.sidebar.title("📂 Data Bank Source")
uploaded_file = st.sidebar.file_uploader("Upload Custom Excel Data Bank (.xlsx)", type=["xlsx"])
default_path = DIR_FILE if os.path.exists(DIR_FILE) else RAW_FILE

if uploaded_file is not None:
    mat, lay, sac, tl, X, Y, model, bench_df = load_and_train(uploaded_file)
elif os.path.exists(default_path):
    mat, lay, sac, tl, X, Y, model, bench_df = load_and_train(default_path)
else:
    st.error(f"Could not find `{RAW_FILE}` in the current folder. Please upload an Excel file in the sidebar.")
    st.stop()

st.sidebar.success(f"Loaded {len(mat)} records | {len(lay)} layers | {mat.customer_name.nunique()} customers")
st.sidebar.info("**Orientation Convention:**\n`[RCV] Left` = Receiver Side\n`Right [SRC]` = Source Side (Incident Sound)")

def get_record(material_id):
    s = mat[mat.material_id == material_id].iloc[0]
    layers = lay[lay.material_id == material_id].sort_values("layer_no").drop(columns="material_id")
    def long(df, prefix):
        r = df[df.material_id == material_id].drop(columns="material_id").iloc[0]
        r = r[[c for c in r.index if c.startswith(prefix)]]
        return pd.DataFrame({"freq_Hz": [int(c.split("_")[1]) for c in r.index], prefix.strip("_"): r.values})
    curves = long(sac, "SAC_").merge(long(tl, "TL_")).merge(long(tl, "IL_"))
    return s, layers, curves

# ==============================================================================
# MAIN UI: 4 INTERACTIVE TABS
# ==============================================================================
st.title("🔊 Automotive NVH Acoustic Material Data Bank & Predictor")
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Recall & Inspect",
    "⚖️ 2. Compare Re-Tests & Records",
    "🔮 3. Predict New Stack",
    "📊 4. Model Accuracy & Data Export"
])

# ------------------------------------------------------------------------------
# TAB 1: RECALL & INSPECT (STEPS 2 & 3)
# ------------------------------------------------------------------------------
with tab1:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        cust_q = st.text_input("Search Customer (partial, case-insensitive)", value="")
    with col_f2:
        mat_q = st.text_input("Search Material Name (partial, case-insensitive)", value="")

    hits = mat[mat.customer_name.str.contains(cust_q, case=False, na=False)]
    if mat_q:
        hits = hits[hits.material_name.str.contains(mat_q, case=False, na=False)]
    hits = hits.sort_values("date_of_evaluation", ascending=False)

    st.subheader(f"Matching Test Records ({len(hits)})")
    st.dataframe(
        hits[["material_id", "customer_name", "material_name", "iteration",
              "receiver_side_material", "source_side_material", "layer_config",
              "total_thickness_mm", "total_gsm", "nrc", "alpha_w", "stc", "date_of_evaluation"]],
        use_container_width=True, hide_index=True
    )

    if len(hits) > 0:
        sel_mid = st.selectbox(
            "Select a Test Record to Inspect Layers & Curves:",
            options=hits["material_id"].tolist(),
            format_func=lambda m: f"{m} | {mat.loc[mat.material_id==m, 'customer_name'].values[0]} - {mat.loc[mat.material_id==m, 'material_name'].values[0]} (iter {mat.loc[mat.material_id==m, 'iteration'].values[0]})"
        )
        s, layers_df, c = get_record(sel_mid)
        st.markdown(f"**Layer Stack (`[RCV] Left <-- Right [SRC]`):** `{s.layer_config}`")
        st.dataframe(layers_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        titles = [f"SAC (NRC = {s.nrc:.2f})", f"TL (STC = {int(s.stc)} dB)", "IL (dB)"]
        for a, y_col, t in zip(ax, ["SAC", "TL", "IL"], titles):
            a.plot(c.freq_Hz, c[y_col], "o-", lw=2)
            a.set_title(t); a.set_ylabel(y_col)
            format_freq_axis(a)
        fig.suptitle(f"{sel_mid}: {s.layer_config}", fontsize=10, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------------------------------
# TAB 2: COMPARE RE-TESTS OR ANY RECORDS (STEP 4)
# ------------------------------------------------------------------------------
with tab2:
    mode = st.radio("Comparison Mode:", ["Compare Re-Tests of Same Material", "Compare Any Custom Material IDs"], horizontal=True)

    if mode == "Compare Re-Tests of Same Material":
        retested = mat.groupby(["customer_name", "material_name"]).filter(lambda x: len(x) > 1)
        pairs = retested[["customer_name", "material_name"]].drop_duplicates().sort_values(["customer_name", "material_name"])
        pair_list = [f"{r.customer_name} | {r.material_name}" for _, r in pairs.iterrows()]
        sel_pair = st.selectbox("Select Re-Tested Material:", pair_list)
        c_name, m_name = sel_pair.split(" | ", 1)
        it = mat[(mat.customer_name == c_name) & (mat.material_name == m_name)].sort_values("iteration")
        mids_to_plot = it["material_id"].tolist()
        st.dataframe(
            it[["material_id", "iteration", "date_of_evaluation", "layer_config",
                "total_thickness_mm", "total_gsm", "nrc", "alpha_w", "stc"]],
            use_container_width=True, hide_index=True
        )
    else:
        mids_to_plot = st.multiselect(
            "Select Any Material IDs to Overlay:",
            options=mat["material_id"].tolist(),
            default=mat["material_id"].iloc[:2].tolist(),
            format_func=lambda m: f"{m} ({mat.loc[mat.material_id==m, 'customer_name'].values[0]} - {mat.loc[mat.material_id==m, 'material_name'].values[0]})"
        )

    if mids_to_plot:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        for a, test in zip(ax, ["SAC", "TL", "IL"]):
            for mid in mids_to_plot:
                s, _, c = get_record(mid)
                metric_str = f"NRC={s.nrc:.2f}" if test == "SAC" else f"STC={int(s.stc)}"
                a.plot(c.freq_Hz, c[test], "o-", label=f"{mid} (iter {s.iteration}, {metric_str})")
            a.set_title(f"{test} Comparison"); a.set_ylabel(test)
            format_freq_axis(a)
            a.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        if len(mids_to_plot) >= 2:
            first, last = mids_to_plot[0], mids_to_plot[-1]
            s1, _, c1 = get_record(first)
            s2, _, c2 = get_record(last)
            st.markdown(f"**Frequency-by-Frequency Change from `{first}` (NRC={s1.nrc}, STC={s1.stc}) to `{last}` (NRC={s2.nrc}, STC={s2.stc}):**")
            diff = pd.DataFrame({
                "freq_Hz": c1.freq_Hz,
                "SAC_change": (c2.SAC - c1.SAC).round(3),
                "TL_change_dB": (c2.TL - c1.TL).round(1),
                "IL_change_dB": (c2.IL - c1.IL).round(1)
            })
            st.dataframe(diff.set_index("freq_Hz").T, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: PREDICT NEW STACK (STEPS 7 & 8)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("""
    **Layer Ordering Rule (`[RCV] Top Row  <---  Bottom Row [SRC]`):**
    - **First Row (Row 1)** = **RECEIVER SIDE** (Leftmost, Cabin / Backing side)
    - **Last Row (Bottom Row)** = **SOURCE SIDE** (Rightmost, Incident sound wave hits this layer first)
    - Leave property cells as `0` (or default) to automatically use `TYPICAL` properties for that material, or type custom overrides directly in the table!
    """)

    default_stack_df = pd.DataFrame([
        {"layer_material": "PU Foam",        "thickness_mm": 15.0, "density_kg_m3": 50.0,   "flow_resistivity": 16000.0, "perforation_dia_mm": 0.0, "perforation_pct": 0.0},
        {"layer_material": "MLV Barrier",    "thickness_mm": 2.0,  "density_kg_m3": 2000.0, "flow_resistivity": 1e9,     "perforation_dia_mm": 0.0, "perforation_pct": 0.0},
        {"layer_material": "PU Foam",        "thickness_mm": 20.0, "density_kg_m3": 45.0,   "flow_resistivity": 14000.0, "perforation_dia_mm": 0.0, "perforation_pct": 0.0},
        {"layer_material": "Nonwoven Scrim", "thickness_mm": 0.5,  "density_kg_m3": 60.0,   "flow_resistivity": 600.0,   "perforation_dia_mm": 0.0, "perforation_pct": 0.0},
    ])

    edited_df = st.data_editor(
        default_stack_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "layer_material": st.column_config.SelectboxColumn("Layer Material (Top=RCV -> Bottom=SRC)", options=LAYER_TYPES, required=True),
            "thickness_mm": st.column_config.NumberColumn("Thickness (mm)", min_value=0.05, max_value=100.0, step=0.5, required=True),
            "density_kg_m3": st.column_config.NumberColumn("Density (kg/m³) [0=Auto]", min_value=0.0, step=5.0),
            "flow_resistivity": st.column_config.NumberColumn("Flow Resistivity (Pa·s/m²) [0=Auto]", min_value=0.0, step=500.0),
            "perforation_dia_mm": st.column_config.NumberColumn("Perf Dia (mm)", min_value=0.0, max_value=20.0, step=0.5),
            "perforation_pct": st.column_config.NumberColumn("Perf (%)", min_value=0.0, max_value=50.0, step=0.5),
        }
    )

    def build_layers_from_df(df_in):
        stack = []
        warnings_list = []
        for _, r in df_in.iterrows():
            mat_name = r["layer_material"]
            role, sig, por, tort, vl, tl_, dens = TYPICAL[mat_name]
            use_dens = float(r["density_kg_m3"]) if r["density_kg_m3"] and r["density_kg_m3"] > 0 else float(dens)
            use_sig  = float(r["flow_resistivity"]) if r["flow_resistivity"] and r["flow_resistivity"] > 0 else float(sig)
            t_mm     = float(r["thickness_mm"])

            sub_train = lay[lay.layer_material == mat_name]
            if len(sub_train) > 0:
                for col_k, val_k in [("density_kg_m3", use_dens), ("flow_resistivity", use_sig)]:
                    vmin, vmax = sub_train[col_k].min(), sub_train[col_k].max()
                    if not (vmin * 0.7 <= val_k <= vmax * 1.3):
                        warnings_list.append(f"`{mat_name}` {col_k}={val_k:g} is outside training range [{vmin:.1f}, {vmax:.1f}]")

            stack.append(dict(
                layer_material=mat_name, role=role, thickness_mm=t_mm,
                density_kg_m3=use_dens, gsm=use_dens * t_mm,
                flow_resistivity=use_sig, porosity=float(por), tortuosity=float(tort),
                visc_char_len_um=float(vl), therm_char_len_um=float(tl_),
                perforation_dia_mm=float(r["perforation_dia_mm"] or 0.0),
                perforation_pct=float(r["perforation_pct"] or 0.0)
            ))
        return stack, warnings_list

    if len(edited_df) > 0:
        new_stack, warn_msgs = build_layers_from_df(edited_df)
        for w in warn_msgs:
            st.warning(w)

        def run_prediction(layers_list):
            x = pd.DataFrame([featurize(layers_list)])[X.columns]
            y = pd.Series(model.predict(x)[0], index=Y.columns)
            sd = pd.Series(model.predict_std(x)[0], index=Y.columns)
            g = lambda s, p: s[[f"{p}_{f}" for f in FREQ]].values
            sac_arr = np.clip(g(y, "SAC"), 0.01, 0.99)
            tl_arr  = g(y, "TL")
            il_arr  = g(y, "IL")
            cfg_str = "[RCV] " + " | ".join(f"{l['layer_material']} {l['thickness_mm']}mm" for l in layers_list) + " [SRC]"
            return dict(
                sac=sac_arr, tl=tl_arr, il=il_arr,
                sac_sd=g(sd, "SAC"), tl_sd=g(sd, "TL"), il_sd=g(sd, "IL"),
                nrc=calc_nrc(sac_arr), stc=calc_stc(tl_arr), config=cfg_str,
                tot_t=sum(l["thickness_mm"] for l in layers_list),
                tot_gsm=sum(l["gsm"] for l in layers_list)
            )

        p = run_prediction(new_stack)
        p_flip = run_prediction(new_stack[::-1])

        st.markdown(f"**Configured Stack:** `{p['config']}`")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Predicted NRC", f"{p['nrc']:.2f}")
        m2.metric("Predicted STC", f"{p['stc']} dB")
        m3.metric("Total Thickness", f"{p['tot_t']:.1f} mm")
        m4.metric("Total GSM", f"{p['tot_gsm']:.0f} g/m²")
        m5.metric("If Installed Backwards", f"NRC {p_flip['nrc']:.2f} | STC {p_flip['stc']} dB")

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        titles = [f"Predicted SAC (NRC = {p['nrc']:.2f})", f"Predicted TL (STC = {p['stc']} dB)", "Predicted IL (dB)"]
        for a, k, t in zip(ax, ["sac", "tl", "il"], titles):
            y_val, sd_val = p[k], p[k + "_sd"]
            a.plot(FREQ, y_val, "o-", lw=2, label="Ensemble Mean Prediction")
            a.fill_between(FREQ, y_val - sd_val, y_val + sd_val, alpha=0.25, label="±1 SD (Sub-Model Spread)")
            a.set_title(t)
            format_freq_axis(a)
            a.legend(fontsize=8)
        fig.suptitle(p["config"], fontsize=10, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

        result_df = pd.DataFrame({
            "Hz": FREQ,
            "SAC": p["sac"].round(3),
            "SAC_SD": p["sac_sd"].round(3),
            "TL_dB": p["tl"].round(1),
            "TL_SD_dB": p["tl_sd"].round(1),
            "IL_dB": p["il"].round(1),
            "IL_SD_dB": p["il_sd"].round(1),
        })
        st.dataframe(result_df.set_index("Hz").T, use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name="Prediction_Curves", index=False)
            pd.DataFrame(new_stack).to_excel(writer, sheet_name="Configured_Layers", index=False)
        st.download_button(
            label="📥 Download Prediction (.xlsx)",
            data=buf.getvalue(),
            file_name="nvh_prediction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ------------------------------------------------------------------------------
# TAB 4: MODEL ACCURACY & DATA EXPORT (STEPS 1, 5 & 6)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("5-Fold Group Cross-Validation Accuracy Across Candidate Models")
    st.dataframe(bench_df, use_container_width=True, hide_index=True)
    st.info("Active Production Model: **AcousticHybridModel** (30% MLP-ReLU + 30% MLP-Tanh + 25% ExtraTrees + 15% RandomForest with Delany-Bazley & Mass-Law Physics Priors).")