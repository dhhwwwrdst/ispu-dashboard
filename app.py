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
