import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Clustering ISPU",
    page_icon="🌫️",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset_ispu.csv")

# =========================
# LOAD MODEL
# =========================
with open("kmeans_model.pkl","rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# =========================
# FITUR
# =========================
fitur = [
    "pm_sepuluh",
    "pm_duakomalima",
    "sulfur_dioksida",
    "karbon_monoksida",
    "ozon",
    "nitrogen_dioksida"
]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌫️ Dashboard ISPU")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Overview",
        "📊 Dataset Insight",
        "📈 Visualisasi Data",
        "🤖 Analisis Clustering",
        "🔍 Cluster Checker"
    ]
)

# =====================================================
# OVERVIEW
# =====================================================

if menu=="🏠 Overview":

    st.title("🌫️ Dashboard Analisis Clustering ISPU DKI Jakarta")

    st.markdown("""
Dashboard ini menyajikan hasil analisis kualitas udara menggunakan algoritma **K-Means Clustering** berdasarkan enam parameter pencemar udara.
""")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Jumlah Data",
        len(df)
    )

    col2.metric(
        "Parameter",
        len(fitur)
    )

    col3.metric(
        "Jumlah Cluster",
        4
    )

    col4.metric(
        "Kategori ISPU",
        df["kategori"].nunique()
    )

    st.divider()

    st.subheader("Informasi Dataset")

    st.write("""
Dataset berasal dari data ISPU DKI Jakarta tahun 2023–2025 yang telah melalui proses:

- Data Cleaning
- Missing Value Handling
- Normalisasi MinMax
- Clustering K-Means
- Evaluasi Silhouette Score
""")

    st.info(
        "Gunakan menu di sebelah kiri untuk mengeksplorasi dataset dan hasil clustering."
    )

# =====================================================
# DATASET INSIGHT
# =====================================================

elif menu == "📊 Dataset Insight":

    st.title("📊 Dataset Insight")

    st.write("Eksplorasi dataset berdasarkan tahun, bulan, dan stasiun pemantauan.")

    st.divider()

    # ==========================
    # FILTER
    # ==========================

    col1, col2, col3 = st.columns(3)

    tahun = col1.selectbox(
        "Pilih Tahun",
        ["Semua"] + sorted(df["periode_data"].astype(str).unique().tolist())
    )

    bulan = col2.selectbox(
        "Pilih Bulan",
        ["Semua"] + sorted(df["bulan"].astype(str).unique().tolist())
    )

    stasiun = col3.selectbox(
        "Pilih Stasiun",
        ["Semua"] + sorted(df["stasiun"].unique().tolist())
    )

    data = df.copy()

    if tahun != "Semua":
        data = data[data["periode_data"].astype(str) == tahun]

    if bulan != "Semua":
        data = data[data["bulan"].astype(str) == bulan]

    if stasiun != "Semua":
        data = data[data["stasiun"] == stasiun]

    st.divider()

    # ==========================
    # METRIC
    # ==========================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Jumlah Data",
        len(data)
    )

    c2.metric(
        "Jumlah Stasiun",
        data["stasiun"].nunique()
    )

    c3.metric(
        "Kategori ISPU",
        data["kategori"].nunique()
    )

    st.divider()

    st.subheader("Preview Dataset")

    st.dataframe(
        data,
        use_container_width=True,
        height=500
    )

    st.divider()

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        data[fitur].describe().round(2),
        use_container_width=True
    )
