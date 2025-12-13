import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_pupuk_xgboost.pkl")
label_encoder = joblib.load("label_encoder.pkl")
fitur_model = joblib.load("fitur_model.pkl")

st.title("🌱 Sistem Rekomendasi Pupuk")
st.write("Masukkan data tanaman untuk mendapatkan rekomendasi pupuk")

# Input user
jenis_tanaman = st.selectbox(
    "Jenis Tanaman",
    ["padi", "jagung"]
)

usia_tanaman = st.number_input(
    "Usia Tanaman (bulan)",
    min_value=0,
    max_value=24,
    step=1
)

jenis_tanah = st.selectbox(
    "Jenis Tanah",
    ["lempung", "liat", "berpasir"]
)

# Tombol prediksi
if st.button("Dapatkan Rekomendasi"):
    input_user = pd.DataFrame([{
        "jenis_tanaman": jenis_tanaman,
        "usia_tanaman_bulan": usia_tanaman,
        "jenis_tanah": jenis_tanah
    }])

    input_encoded = pd.get_dummies(
        input_user,
        columns=["jenis_tanaman", "jenis_tanah"]
    )

    input_encoded = input_encoded.reindex(
        columns=fitur_model,
        fill_value=0
    )

    prediksi = model.predict(input_encoded)
    hasil = label_encoder.inverse_transform(prediksi)

    st.success(f"✅ Rekomendasi pupuk: **{hasil[0]}**")

st.write("Fitur model:")
st.write(fitur_model)

st.write("Input encoded:")
st.write(input_encoded)
