"""
Hybrid PCA Talent Map with Centroids & 3D PCA

- Hybrid PCA (Skill PCA1 vs Personal PCA1)
- K-Means clustering on skill + personal features
- 2D Hybrid PCA scatter with cluster centroids
- 3D PCA scatter (combined features) with centroids
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_loader import load_data


# -----------------------------
# Column configuration
# -----------------------------

# Saf skill tarafı (bilişsel + teknik + GPA)
SKILL_COLS = [
    "English",
    "Logical",
    "Quant",
    "Domain",
    "ComputerProgramming",
    "collegeGPA",
]

# Kişilik + arka plan tarafı
PERSONAL_COLS = [
    "conscientiousness",
    "agreeableness",
    "extraversion",
    "nueroticism",              # dataset'te yazım bu
    "openess_to_experience",
    "CollegeTier",
    "CollegeCityTier",
]

# -1 kullanılan kolonlar (test yok anlamında)
MISSING_AS_MINUS_ONE_COLS = [
    "Domain",
    "ComputerProgramming",
]


# -----------------------------
# Data cleaning & feature engineering
# -----------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Temel temizlik: -1'leri NaN yap, dtype'ları düzelt."""
    df = df.copy()

    # -1 -> NaN (ölçülmemiş test skorları)
    for col in MISSING_AS_MINUS_ONE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    # Kümelenecek tüm kolonlar gerçekten var mı?
    missing_skill = [c for c in SKILL_COLS if c not in df.columns]
    missing_personal = [c for c in PERSONAL_COLS if c not in df.columns]
    if missing_skill or missing_personal:
        missing_all = missing_skill + missing_personal
        st.error(
            "Dataset'te beklenen bazı kolonlar eksik:\n"
            + ", ".join(missing_all)
            + "\nLütfen CSV kolon isimlerini kontrol et."
        )
        st.stop()

    return df


def compute_hybrid_pca_and_clusters(
    df: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, float, float, float]:
    """
    - Skill tarafı için PCA1
    - Personal tarafı için PCA1
    - Tüm skill+personal özellikleri ile K-Means
    - Silhouette ve PCA varyanslarını döndürür
    """
    df = df.copy()

    # NaN'leri at: PCA ve K-Means NaN kabul etmiyor
    df = df.dropna(subset=SKILL_COLS + PERSONAL_COLS)
    if df.empty or len(df) < k:
        raise ValueError("Bu filtrelerle PCA / clustering için yeterli veri yok.")

    # --- Skill tarafı: Standartlaştırma + PCA1
    scaler_skill = StandardScaler()
    X_skill = scaler_skill.fit_transform(df[SKILL_COLS])

    pca_skill = PCA(n_components=1, random_state=42)
    skill_pc1 = pca_skill.fit_transform(X_skill)[:, 0]
    skill_var = float(pca_skill.explained_variance_ratio_[0])

    # --- Personal tarafı: Standartlaştırma + PCA1
    scaler_personal = StandardScaler()
    X_personal = scaler_personal.fit_transform(df[PERSONAL_COLS])

    pca_personal = PCA(n_components=1, random_state=42)
    personal_pc1 = pca_personal.fit_transform(X_personal)[:, 0]
    personal_var = float(pca_personal.explained_variance_ratio_[0])

    # --- K-Means: Skill + Personal tüm özellikler birlikte
    X_combined = np.concatenate([X_skill, X_personal], axis=1)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_combined)

    sil = silhouette_score(X_combined, clusters)

    # --- Sonuç DataFrame
    df["Skill_PC1"] = skill_pc1
    df["Personal_PC1"] = personal_pc1
    df["Cluster"] = clusters

    return df, sil, skill_var, personal_var


# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(
        page_title="Hybrid PCA Talent Map",
        page_icon="🧬",
        layout="wide",
    )

    df = load_data()
    df = clean_data(df)

    st.title("🧬 Hybrid PCA Talent Map")
    st.markdown(
        """
This page analyzes graduates using **two different PCA axes** and **K-Means clustering**:

- **X-axis – Skill PCA1**: Summarizes cognitive + technical abilities (English, Logical, 
  Quantitative, Domain, Programming, and GPA) on a single axis.
- **Y-axis – Personal PCA1**: Summarizes personality traits (Big Five), college tier, 
  and city tier on a single axis.

Additionally:
- The 2D Hybrid PCA chart displays **cluster centroids**.
- A **3D PCA scatter plot** computed over all skill + personal features visualizes 
  cluster separation in space.
"""
    )

    # ----------------- Filters & settings -----------------
    with st.expander("🔧 Filters & Settings", expanded=True):
        c1, c2 = st.columns(2)

        with c1:
            if "Degree" in df.columns:
                degree_opts = sorted(df["Degree"].dropna().unique())
                selected_degree = st.multiselect(
                    "Degree",
                    degree_opts,
                    default=degree_opts,
                )
            else:
                selected_degree = None

            if "CollegeTier" in df.columns:
                tier_opts = sorted(df["CollegeTier"].dropna().unique())
                selected_tier = st.multiselect(
                    "College Tier",
                    tier_opts,
                    default=tier_opts,
                )
            else:
                selected_tier = None

        with c2:
            k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)

            # color_by removed since Salary is gone. Defaulting to Cluster.
            color_by = "Cluster"

    # Filtreleri uygula
    mask = pd.Series(True, index=df.index)

    if selected_degree is not None:
        mask &= df["Degree"].isin(selected_degree)

    if selected_tier is not None:
        mask &= df["CollegeTier"].isin(selected_tier)

    df_filtered = df[mask]

    # ----------------- PCA + Clustering -----------------
    try:
        df_hybrid, sil, skill_var, personal_var = compute_hybrid_pca_and_clusters(
            df_filtered, k
        )
    except ValueError as e:
        st.warning(str(e))
        return

    # ----------------- Metrics -----------------
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", f"{len(df_hybrid):,}")
    m2.metric("Clusters", k)
    m3.metric("Silhouette Score", f"{sil:.3f}")
    m4.metric("Skill / Personal PCA Var", f"{skill_var*100:.1f}% / {personal_var*100:.1f}%")

    # ----------------- Color arg (hem 2D hem 3D için) -----------------
    # ----------------- Color arg (hem 2D hem 3D için) -----------------
    # Always color by Cluster now
    color_arg = df_hybrid["Cluster"].astype(str)
    color_continuous = None

    # ----------------- 2D Hybrid PCA scatter + centroids -----------------
    st.subheader("🎯 Hybrid PCA Visualization (Skill PCA1 × Personal PCA1) + Centroids")

    hover_cols = []
    hover_cols = []
    for col in ["Degree", "Specialization", "collegeGPA"]:
        if col in df_hybrid.columns:
            hover_cols.append(col)

    fig_2d = px.scatter(
        df_hybrid,
        x="Skill_PC1",
        y="Personal_PC1",
        color=color_arg,
        color_continuous_scale=color_continuous,
        hover_data=hover_cols,
        labels={
            "Skill_PC1": "Skill PCA1 (Cognitive & Technical Skills)",
            "Personal_PC1": "Personal PCA1 (Personality & Background)",
            "color": color_by,
        },
        title=f"Hybrid PCA Talent Map (K = {k})",
    )

    # Centroid hesapla (2D)
    centroids_2d = (
        df_hybrid.groupby("Cluster")[["Skill_PC1", "Personal_PC1"]]
        .mean()
        .reset_index()
    )

    # Centroidleri ekle
    fig_2d.add_scatter(
        x=centroids_2d["Skill_PC1"],
        y=centroids_2d["Personal_PC1"],
        mode="markers+text",
        marker=dict(
            size=16,
            symbol="x",
            color="white",
            line=dict(width=2, color="black"),
        ),
        text=centroids_2d["Cluster"].astype(str),
        textposition="top center",
        name="Cluster Centroid",
        showlegend=True,
    )

    fig_2d.update_layout(
        height=650,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend_title_text=color_by,
        modebar=dict(
            bgcolor="rgba(0,0,0,0.3)",
            color="white",
            activecolor="#00D9FF",
        ),
    )

    config_2d = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "hybrid_pca_talent_map_2d",
            "height": 1080,
            "width": 1920,
            "scale": 2,
        },
    }

    st.plotly_chart(fig_2d, use_container_width=True, config=config_2d)

    # ----------------- 3D PCA scatter + centroids -----------------
    st.subheader("🧊 3D PCA Visualization (Combined Skill + Personal Features)")

    # 3D PCA için feature matrisi (skill + personal)
    feature_cols = SKILL_COLS + PERSONAL_COLS
    df_3d = df_hybrid.dropna(subset=feature_cols).copy()  # zaten temiz ama garanti olsun

    scaler_3d = StandardScaler()
    X_3d = scaler_3d.fit_transform(df_3d[feature_cols])

    pca_3d = PCA(n_components=3, random_state=42)
    coords_3d = pca_3d.fit_transform(X_3d)

    df_3d["PC3D_1"] = coords_3d[:, 0]
    df_3d["PC3D_2"] = coords_3d[:, 1]
    df_3d["PC3D_3"] = coords_3d[:, 2]

    # 3D centroidler
    centroids_3d = (
        df_3d.groupby("Cluster")[["PC3D_1", "PC3D_2", "PC3D_3"]]
        .mean()
        .reset_index()
    )

    # 3D renk serisi (Cluster ise str, diğer durumda numeric)
    # 3D renk serisi
    color_arg_3d = df_3d["Cluster"].astype(str)
    color_scale_3d = None

    fig_3d = px.scatter_3d(
        df_3d,
        x="PC3D_1",
        y="PC3D_2",
        z="PC3D_3",
        color=color_arg_3d,
        color_continuous_scale=color_scale_3d,
        hover_data=hover_cols,
        labels={
            "PC3D_1": "PC1 (Combined)",
            "PC3D_2": "PC2 (Combined)",
            "PC3D_3": "PC3 (Combined)",
            "color": color_by,
        },
        title="3D PCA Scatter of Skill + Personal Features",
    )

    # 3D centroid scatter
    fig_3d.add_trace(
        go.Scatter3d(
            x=centroids_3d["PC3D_1"],
            y=centroids_3d["PC3D_2"],
            z=centroids_3d["PC3D_3"],
            mode="markers+text",
            marker=dict(
                size=8,
                symbol="x",
                color="white",
                line=dict(width=2, color="black"),
            ),
            text=centroids_3d["Cluster"].astype(str),
            textposition="top center",
            name="Cluster Centroid",
        )
    )

    fig_3d.update_layout(
        height=700,
        paper_bgcolor="#0E1117",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        font=dict(color="#FAFAFA"),
        legend_title_text=color_by,
    )

    config_3d = {
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "hybrid_pca_talent_map_3d",
            "height": 1080,
            "width": 1920,
            "scale": 2,
        },
    }

    st.plotly_chart(fig_3d, use_container_width=True, config=config_3d)

    # ----------------- Cluster profiles table -----------------
    st.divider()
    st.subheader("📊 Cluster Profiles")

    agg_dict = {
        "Skill_PC1": ["mean"],
        "Personal_PC1": ["mean"],
    }
    for col in SKILL_COLS:
        agg_dict[col] = ["mean"]

    cluster_stats = df_hybrid.groupby("Cluster").agg(agg_dict).round(2)
    cluster_stats.columns = ["_".join(map(str, c)).strip("_") for c in cluster_stats.columns]
    cluster_stats = cluster_stats.reset_index()

    st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

    # ----------------- Auto-generated key findings -----------------
    st.subheader("🧾 Key Findings (auto-generated hints)")

    lines = []
    for _, row in cluster_stats.iterrows():
        cid = int(row["Cluster"])
        skill_pc = row["Skill_PC1_mean"]
        pers_pc = row["Personal_PC1_mean"]

        # Skill level interpretation
        if skill_pc >= 0:
            skill_text = "skill level ABOVE average"
        else:
            skill_text = "skill level BELOW average"

        # Personal interpretation
        if pers_pc >= 0:
            personal_text = "personality/background profile leans POSITIVE on PCA1 axis"
        else:
            personal_text = "personality/background profile leans NEGATIVE on PCA1 axis"

        line = (
            f"- **Cluster {cid}** → {skill_text}, {personal_text}."
        )
        lines.append(line)

    st.markdown("\n".join(lines))

    st.caption(
        """
**Notes:**

- Skill PCA1 and Personal PCA1 axes should be interpreted directionally; **0** represents the average profile of the dataset.
- In the 2D chart, centroids show the "center" position of clusters; cluster assignment is based on the nearest centroid.
- 3D PCA reduces all skill + personal features to 3 dimensions simultaneously. Cluster boundaries are more distinct in this view compared to 2D.
- Silhouette Score values around ~0.1–0.2 indicate that clusters are not fully separated but still offer meaningful segments.
"""
    )


if __name__ == "__main__":
    main()
