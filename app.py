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
df["tahun"] = df["periode_data"].astype(str).str[:4]
df["bulan_num"] = df["periode_data"].astype(str).str[4:6]
bulan_map = {
    "01":"Januari",
    "02":"Februari",
    "03":"Maret",
    "04":"April",
    "05":"Mei",
    "06":"Juni",
    "07":"Juli",
    "08":"Agustus",
    "09":"September",
    "10":"Oktober",
    "11":"November",
    "12":"Desember"
}

df["nama_bulan"] = df["bulan_num"].map(bulan_map)

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

    df["tahun"] = df["periode_data"].astype(str).str[:4]

tahun = col1.selectbox(
    "Pilih Tahun",
    ["Semua"] + sorted(df["tahun"].unique().tolist())
)

    bulan = col2.selectbox(
    "Pilih Bulan",
    ["Semua"] + list(bulan_map.values())
)

    stasiun = col3.selectbox(
        "Pilih Stasiun",
        ["Semua"] + sorted(df["stasiun"].unique().tolist())
    )

    data = df.copy()

    if tahun != "Semua":
    data = data[data["tahun"] == tahun]

    if bulan != "Semua":
    data = data[data["nama_bulan"] == bulan]

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

# =====================================================
# VISUALISASI DATA
# =====================================================

elif menu == "📈 Visualisasi Data":

    st.title("📈 Visualisasi Data")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Heatmap",
        "📊 Distribusi",
        "🥧 Proporsi",
        "🗺️ PCA"
    ])

    with tab1:

        st.subheader("Heatmap Korelasi Parameter Polutan")

        corr = df[fitur].corr()

        fig, ax = plt.subplots(figsize=(8,6))

        sns.heatmap(
            corr,
            annot=True,
            cmap="YlGnBu",
            ax=ax
        )

        st.pyplot(fig)

    with tab2:

        pilihan = st.selectbox(
            "Pilih Visualisasi",
            [
                "Distribusi Tahun",
                "Distribusi Bulan",
                "Distribusi Stasiun",
                "Distribusi Kategori"
            ]
        )

        if pilihan=="Distribusi Tahun":

            fig = px.histogram(
                df,
                x="periode_data"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        elif pilihan=="Distribusi Bulan":

            fig = px.histogram(
                df,
                x="bulan"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        elif pilihan=="Distribusi Stasiun":

            jumlah = df["stasiun"].value_counts()

            fig = px.bar(
                jumlah,
                x=jumlah.index,
                y=jumlah.values
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            jumlah = df["kategori"].value_counts()

            fig = px.bar(
                jumlah,
                x=jumlah.index,
                y=jumlah.values
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab3:

        st.subheader("Proporsi Kategori ISPU")

        fig = px.pie(
            df,
            names="kategori"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab4:

        st.subheader("Visualisasi PCA")

        X = scaler.transform(df[fitur])

        pca = PCA(n_components=2)

        hasil = pca.fit_transform(X)

        pca_df = pd.DataFrame()

        pca_df["PC1"] = hasil[:,0]
        pca_df["PC2"] = hasil[:,1]
        pca_df["Cluster"] = df["cluster"].astype(str)

        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =====================================================
# ANALISIS CLUSTERING
# =====================================================

elif menu == "🤖 Analisis Clustering":

    st.title("🤖 Analisis Clustering K-Means")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Elbow",
        "📈 Silhouette",
        "🔥 Karakteristik Cluster",
        "🌫️ Distribusi Risiko"
    ])

    with tab1:

        st.subheader("Metode Elbow")

        inertia = []

        X = scaler.transform(df[fitur])

        for k in range(2,11):

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            model.fit(X)

            inertia.append(model.inertia_)

        fig = px.line(
            x=list(range(2,11)),
            y=inertia,
            markers=True
        )

        fig.update_layout(
            xaxis_title="Jumlah Cluster",
            yaxis_title="Inertia"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.success("Jumlah cluster optimal diperoleh pada k = 4.")

    with tab2:

        st.subheader("Silhouette Score")

        nilai = {
            2:0.2424,
            3:0.2373,
            4:0.2553,
            5:0.2321,
            6:0.2203,
            7:0.2245,
            8:0.2347,
            9:0.2273,
            10:0.2292
        }

        fig = px.line(
            x=list(nilai.keys()),
            y=list(nilai.values()),
            markers=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.metric(
            "Silhouette Score Terbaik",
            "0.2553"
        )

        st.info(
            "Cluster optimal berada pada k = 4."
        )

    with tab3:

        st.subheader("Karakteristik Rata-rata Polutan")

        cluster_summary = df.groupby("cluster")[fitur].mean().round(2)

        fig, ax = plt.subplots(figsize=(10,6))

        sns.heatmap(
            cluster_summary,
            annot=True,
            cmap="YlOrRd",
            fmt=".1f",
            ax=ax
        )

        st.pyplot(fig)

        st.dataframe(cluster_summary)

    with tab4:

        st.subheader("Distribusi Data per Cluster")

        jumlah = df["cluster"].value_counts().sort_index()

        fig = px.bar(
            x=jumlah.index.astype(str),
            y=jumlah.values,
            labels={
                "x":"Cluster",
                "y":"Jumlah Data"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            jumlah.rename("Jumlah Data")
        )

# =====================================================
# CLUSTER CHECKER
# =====================================================

elif menu == "🔍 Cluster Checker":

    st.title("🔍 Cluster Checker")

    st.write(
        "Masukkan nilai parameter polutan untuk mengetahui cluster dan kategori risiko berdasarkan model K-Means."
    )

    col1, col2 = st.columns(2)

    pm10 = col1.number_input("PM10", min_value=0.0)
    pm25 = col1.number_input("PM2.5", min_value=0.0)
    so2 = col1.number_input("SO₂", min_value=0.0)

    co = col2.number_input("CO", min_value=0.0)
    o3 = col2.number_input("O₃", min_value=0.0)
    no2 = col2.number_input("NO₂", min_value=0.0)

    if st.button("Cek Cluster"):

        input_df = pd.DataFrame([[
            pm10,
            pm25,
            so2,
            co,
            o3,
            no2
        ]], columns=fitur)

        data_scaled = scaler.transform(input_df)

        cluster = kmeans.predict(data_scaled)[0]

        ranking = (
            df.groupby("cluster")[fitur]
            .mean()
            .mean(axis=1)
            .sort_values()
        )

        label = {
            ranking.index[0]: "🟢 Risiko Rendah",
            ranking.index[1]: "🟡 Risiko Sedang",
            ranking.index[2]: "🟠 Risiko Tinggi",
            ranking.index[3]: "🔴 Risiko Sangat Tinggi"
        }

        st.success(f"Cluster : {cluster}")
        st.info(f"Kategori Risiko : {label[cluster]}")
