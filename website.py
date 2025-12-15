import streamlit as st
import pandas as pd
import joblib

# Load model & tools
model = joblib.load("model_pupuk.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🌱 Sistem Rekomendasi Pupuk")
st.write("Masukkan kondisi lahan dan tanaman")

# Input user
jenis_tanaman = st.selectbox(
    "Jenis Tanaman",
    ["padi", "jagung"]
)

jenis_tanah = st.selectbox(
    "Jenis Tanah",
    ["lempung", "liat", "berpasir"]
)

usia = st.number_input(
    "Usia Tanaman (bulan)",
    min_value=0.0,
    max_value=24.0,
    step=0.1
)

ph = st.number_input(
    "pH Tanah",
    min_value=4.5,
    max_value=7.5,
    step=0.1
)

curah_hujan = st.number_input("Curah Hujan (mm/tahun)")

n_tanah = st.number_input("Nitrogen Tanah (N)")
p_tanah = st.number_input("Fosfor Tanah (P)")
k_tanah = st.number_input("Kalium Tanah (K)")

# Tombol prediksi
if st.button("🔍 Rekomendasikan Pupuk"):

    input_user = pd.DataFrame([{
        "jenis_tanaman": jenis_tanaman,
        "usia_tanaman_bulan": usia,
        "jenis_tanah": jenis_tanah,
        "ph_tanah": ph,
        "curah_hujan_mm": curah_hujan,
        "n_tanah": n_tanah,
        "p_tanah": p_tanah,
        "k_tanah": k_tanah
    }])

    input_encoded = pd.get_dummies(
        input_user,
        columns=["jenis_tanaman", "jenis_tanah"],
        dtype=int
    )

    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    pred = model.predict(input_encoded)
    hasil = label_encoder.inverse_transform(pred)

    st.success(f"👉 Rekomendasi pupuk: **{hasil[0]}**")
