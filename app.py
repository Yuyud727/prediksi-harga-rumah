import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import fitur  # noqa: F401  (pipeline yang disimpan merujuk ke fitur.rekayasa_fitur)

BASE = Path(__file__).parent

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="centered")


@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model_harga_rumah.joblib")


@st.cache_data
def muat_meta():
    return json.loads((BASE / "meta_model.json").read_text(encoding="utf-8"))


def format_rupiah(x):
    if x >= 1e9:
        return f"Rp {x / 1e9:,.2f} Miliar".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {x / 1e6:,.0f} Juta".replace(",", ".")


model = muat_model()
meta = muat_meta()
r, d = meta["rentang"], meta["default"]

st.title("🏠 Prediksi Harga Rumah")
st.caption("Estimasi harga rumah tapak berdasarkan data listing. Hasil adalah perkiraan, bukan penilaian resmi.")

with st.form("form_rumah"):
    kecamatan = st.selectbox("Kecamatan / area", meta["kecamatan"], index=meta["kecamatan"].index("Cinere") if "Cinere" in meta["kecamatan"] else 0)

    c1, c2, c3 = st.columns(3)
    kt = c1.number_input("Kamar tidur", r["kamar_tidur"][0], r["kamar_tidur"][1], d["kamar_tidur"], step=1)
    km = c2.number_input("Kamar mandi", r["kamar_mandi"][0], r["kamar_mandi"][1], d["kamar_mandi"], step=1)
    garasi = c3.number_input("Garasi (mobil)", r["garasi"][0], r["garasi"][1], d["garasi"], step=1)

    c4, c5 = st.columns(2)
    lt = c4.number_input("Luas tanah (m²)", r["luas_tanah"][0], r["luas_tanah"][1], d["luas_tanah"], step=5)
    lb = c5.number_input("Luas bangunan (m²)", r["luas_bangunan"][0], r["luas_bangunan"][1], d["luas_bangunan"], step=5)
    lb_tidak_tahu = st.checkbox("Luas bangunan tidak diketahui")

    kirim = st.form_submit_button("Prediksi harga", type="primary")

if kirim:
    data = pd.DataFrame([{
        "Kamar_Tidur": float(kt),
        "Kamar_Mandi": float(km),
        "Garasi": float(garasi),
        "Luas_Tanah": float(lt),
        "Luas_Bangunan": np.nan if lb_tidak_tahu else float(lb),
        "Kecamatan": kecamatan,
    }])
    harga = float(model.predict(data)[0])
    mdape = meta["metrik_test"]["MdAPE (%)"] / 100

    st.subheader(format_rupiah(harga))
    st.write(f"Perkiraan rentang: **{format_rupiah(harga * (1 - mdape))}** sampai **{format_rupiah(harga * (1 + mdape))}**")
    st.caption(f"Rentang memakai median galat pada data uji (±{mdape * 100:.0f}%). Galat bisa lebih besar untuk rumah mewah atau data yang tidak biasa.")

    if not lb_tidak_tahu and lb > lt * 6:
        st.warning("Luas bangunan tampak tidak wajar dibanding luas tanah. Periksa kembali input.")

with st.expander("Tentang model"):
    m = meta["metrik_test"]
    st.write(f"Model: **{meta['model']}** (dilatih pada {meta['jumlah_data_latih']:,} listing).")
    st.write(f"Evaluasi test set: MAE Rp {m['MAE (Rp)'] / 1e6:,.0f} juta, MAPE {m['MAPE (%)']:.1f}%, MdAPE {m['MdAPE (%)']:.1f}%, R² {m['R2']:.2f}.")
    st.write("Data: hasil scraping listing rumah. Harga listing adalah harga penawaran, bukan harga transaksi.")
