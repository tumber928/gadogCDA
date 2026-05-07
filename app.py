"""
Aplikasi Prediksi Harga Rumah per Unit Area
Menggunakan model AdaBoost hasil training dari Orange Data Mining
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# ─────────────────────────────────────────────
# KONFIGURASI PATH MODEL
# Gunakan path relatif agar kompatibel dengan Streamlit Cloud
# ─────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model_orange.pickle"

# ─────────────────────────────────────────────
# KONFIGURASI FITUR
# Nama fitur harus persis sama dengan nama kolom saat training di Orange
# ─────────────────────────────────────────────
FEATURE_CONFIG = {
    "X1 transaction date": {
        "label": "📅 X1 – Transaction Date (Year)",
        "type": "numeric",
        "input": "number",
        "min": 2010.0,
        "max": 2015.0,
        "default": 2013.0,
        "step": 0.083,   # ~1 bulan dalam desimal tahun
        "format": "%.3f",
        "help": "Tahun transaksi dalam format desimal (cth: 2013.250 = Maret 2013)"
    },
    "X2 house age": {
        "label": "🏠 X2 – House Age (years)",
        "type": "numeric",
        "input": "slider",
        "min": 0.0,
        "max": 50.0,
        "default": 10.0,
        "step": 0.1,
        "help": "Usia bangunan dalam tahun"
    },
    "X3 distance to the nearest MRT station": {
        "label": "🚇 X3 – Distance to Nearest MRT (meters)",
        "type": "numeric",
        "input": "number",
        "min": 0.0,
        "max": 7000.0,
        "default": 500.0,
        "step": 1.0,
        "format": "%.1f",
        "help": "Jarak ke stasiun MRT terdekat dalam meter"
    },
    "X4 number of convenience stores": {
        "label": "🏪 X4 – Number of Convenience Stores",
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "step": 1,
        "help": "Jumlah minimarket/convenience store di sekitar properti"
    },
    "X5 latitude": {
        "label": "🌐 X5 – Latitude",
        "type": "numeric",
        "input": "number",
        "min": 24.9,
        "max": 25.1,
        "default": 24.97,
        "step": 0.0001,
        "format": "%.4f",
        "help": "Koordinat lintang lokasi properti (area Taiwan)"
    },
    "X6 longitude": {
        "label": "🌐 X6 – Longitude",
        "type": "numeric",
        "input": "number",
        "min": 121.4,
        "max": 121.7,
        "default": 121.53,
        "step": 0.0001,
        "format": "%.4f",
        "help": "Koordinat bujur lokasi properti (area Taiwan)"
    },
}

TARGET_NAME = "Y house price of unit area"


# ─────────────────────────────────────────────
# LOAD MODEL (di-cache agar tidak reload setiap interaksi)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Memuat model dari file pickle. Mengembalikan (model, error_message)."""
    if not MODEL_PATH.exists():
        return None, (
            f"❌ File model tidak ditemukan: `{MODEL_PATH.name}`\n\n"
            "Pastikan file `model_orange.pickle` sudah di-upload ke GitHub repository "
            "yang sama dengan `app.py`."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"❌ Gagal memuat model: {e}"


# ─────────────────────────────────────────────
# PREDIKSI DENGAN ORANGE FALLBACK
# ─────────────────────────────────────────────
def predict_with_model(model, input_df: pd.DataFrame):
    """
    Mencoba prediksi dengan scikit-learn interface terlebih dahulu.
    Jika gagal, menggunakan Orange3 native interface.
    Mengembalikan (hasil_prediksi, probabilitas_atau_None, error_message_atau_None)
    """
    # ── Pendekatan 1: scikit-learn-like ──────────────────────────────────
    try:
        prediction = model.predict(input_df)
        return float(prediction[0]), None, None
    except Exception as sklearn_err:
        pass  # coba fallback Orange

    # ── Pendekatan 2: Orange native ───────────────────────────────────────
    try:
        import Orange.data as od

        feature_names = list(input_df.columns)
        row = input_df.iloc[0]

        attrs = []
        for feat in feature_names:
            cfg = FEATURE_CONFIG[feat]
            if cfg["type"] == "numeric":
                attrs.append(od.ContinuousVariable(feat))
            else:
                attrs.append(od.DiscreteVariable(feat, values=cfg["options"]))

        class_var = od.ContinuousVariable(TARGET_NAME)
        domain = od.Domain(attrs, class_vars=class_var)

        # Susun nilai fitur
        X_vals = []
        for feat in feature_names:
            cfg = FEATURE_CONFIG[feat]
            val = row[feat]
            if cfg["type"] == "categorical":
                val = cfg["options"].index(str(val))
            X_vals.append(float(val))

        X_array = np.array([X_vals])
        Y_array = np.full((1, 1), np.nan)

        orange_table = od.Table.from_numpy(domain, X_array, Y_array)
        result = model(orange_table)

        pred_val = float(result[0]) if hasattr(result, "__len__") else float(result)
        return pred_val, None, None

    except ImportError:
        return None, None, (
            "❌ Library `orange3` tidak tersedia.\n\n"
            "Pastikan `orange3` ada di `requirements.txt` dan sudah terinstall "
            "di environment Streamlit Cloud."
        )
    except Exception as orange_err:
        return None, None, (
            f"❌ Prediksi gagal.\n\n"
            f"**Scikit-learn error:** {sklearn_err}\n\n"
            f"**Orange error:** {orange_err}\n\n"
            "Kemungkinan nama kolom input tidak cocok dengan fitur saat training. "
            "Periksa kembali `FEATURE_CONFIG` di `app.py`."
        )


# ─────────────────────────────────────────────
# UI UTAMA
# ─────────────────────────────────────────────
def main():
    # ── Page config ───────────────────────────
    st.set_page_config(
        page_title="Prediksi Harga Rumah",
        page_icon="🏡",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.title("ℹ️ Panduan Penggunaan")
        st.markdown(
            """
            1. Isi semua nilai input di form utama.
            2. Klik tombol **Prediksi** untuk mendapatkan estimasi harga.
            3. Hasil ditampilkan dalam satuan **10.000 NTD / m²** (New Taiwan Dollar).

            ---
            **Dataset:** Real Estate Valuation – Taiwan  
            **Model:** AdaBoost (Orange Data Mining)  
            **File model:** `model_orange.pickle` di GitHub repository ini.
            ---
            > ⚠️ Pastikan file `model_orange.pickle` sudah di-upload ke repo GitHub
            > dan memiliki nama yang sama persis.
            """
        )
        st.divider()
        st.caption("Dibuat dengan Streamlit + Orange3")

    # ── Header ────────────────────────────────
    st.title("🏡 Aplikasi Prediksi Harga Rumah per Unit Area")
    st.markdown(
        "Aplikasi ini menggunakan model machine learning hasil training dari **Orange Data Mining** "
        "dan dijalankan melalui **Streamlit Cloud**."
    )
    st.divider()

    # ── Load model ────────────────────────────
    model, model_err = load_model()

    if model_err:
        st.error(model_err)
        st.stop()

    st.success("✅ Model berhasil dimuat.")

    # ── Form input ────────────────────────────
    st.subheader("📋 Masukkan Data Properti")

    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}

        for i, (feat, cfg) in enumerate(FEATURE_CONFIG.items()):
            col = cols[i % 2]
            with col:
                label = cfg["label"]
                help_text = cfg.get("help", "")

                if cfg["type"] == "numeric":
                    if cfg["input"] == "slider":
                        val = st.slider(
                            label,
                            min_value=float(cfg["min"]),
                            max_value=float(cfg["max"]),
                            value=float(cfg["default"]),
                            step=float(cfg.get("step", 1.0)),
                            help=help_text,
                        )
                    else:  # number_input
                        val = st.number_input(
                            label,
                            min_value=float(cfg["min"]),
                            max_value=float(cfg["max"]),
                            value=float(cfg["default"]),
                            step=float(cfg.get("step", 1.0)),
                            format=cfg.get("format", "%.2f"),
                            help=help_text,
                        )
                else:  # categorical
                    val = st.selectbox(
                        label,
                        options=cfg["options"],
                        help=help_text,
                    )

                input_data[feat] = val

        submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

    # ── Hasil prediksi ────────────────────────
    if submitted:
        input_df = pd.DataFrame([input_data], columns=list(FEATURE_CONFIG.keys()))

        st.subheader("📊 Data Input")
        display_df = input_df.copy()
        display_df.columns = [FEATURE_CONFIG[c]["label"] for c in display_df.columns]
        st.dataframe(display_df.T.rename(columns={0: "Nilai"}), use_container_width=True)

        with st.spinner("Menjalankan prediksi..."):
            pred, proba, err = predict_with_model(model, input_df)

        if err:
            st.error(err)
        else:
            st.divider()
            st.subheader("🎯 Hasil Prediksi")

            col1, col2 = st.columns(2)
            with col1:
                st.success(
                    f"**Estimasi Harga:** `{pred:.2f}` × 10.000 NTD/m²\n\n"
                    f"≈ **{pred * 10000:,.0f} NTD/m²**"
                )
            with col2:
                st.info(
                    f"💡 Harga estimasi ini merupakan **harga per unit area (m²)** "
                    f"dalam satuan 10.000 New Taiwan Dollar (NTD)."
                )

            if proba is not None:
                st.write("**Confidence / Probabilitas:**", proba)


if __name__ == "__main__":
    main()
