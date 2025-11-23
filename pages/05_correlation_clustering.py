"""
GPA & Skills Clustering Analysis

Performs clustered correlation analysis on GPA, skills, academic scores
and personality traits. Hierarchical clustering groups related features
together to reveal natural feature blocks (academic, aptitude, technical,
personality).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from data_loader import load_data


def compute_clustered_order(corr_matrix: pd.DataFrame) -> list[str]:
    """Reorder correlation matrix using hierarchical clustering."""
    # Distance = 1 - correlation (clipped for safety)
    distance_matrix = 1 - corr_matrix.clip(-1, 1)

    # Diagonal mutlaka 0 olmalı (kendisiyle mesafe)
    for i in range(len(distance_matrix)):
        distance_matrix.iloc[i, i] = 0

    # squareform -> upper triangle vektörüne çevir
    dists = squareform(distance_matrix, checks=False)

    # Ward linkage ile hierarchical clustering
    linkage_matrix = hierarchy.linkage(dists, method="ward")
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)

    # Dendrogram yaprak sırasına göre kolonları yeniden sırala
    return [corr_matrix.columns[i] for i in dendro["leaves"]]


def main():
    st.set_page_config(
        page_title="GPA & Skills Clustering",
        page_icon="🧬",
        layout="wide",
    )

    df = load_data()

    st.title("🧬 GPA & Skills Clustering Analysis")
    st.markdown(
        """
        Clustered correlation heatmap for **academic scores, aptitude tests,
        technical skills and personality traits**.

        Hierarchical clustering groups related features together so that:
        - Academic scores (10/12/GPA) form one block  
        - Aptitude scores (English/Logical/Quant) form another block  
        - Technical skills and personality traits connect these blocks
        """
    )

    # ───────────────────────── Filters ─────────────────────────
    with st.expander("🔧 Filters & Feature Selection", expanded=True):
        col1, col2 = st.columns(2)

        # ---- Left: Degree & Specialization filters ----
        with col1:
            degree_opts = sorted(df["Degree"].dropna().unique())
            selected_degree = st.multiselect(
                "Degree",
                options=degree_opts,
                default=degree_opts,
            )

            top_specs = df["Specialization"].value_counts().head(10).index.tolist()
            all_specs = sorted(df["Specialization"].dropna().unique())
            selected_specs = st.multiselect(
                "Specialization",
                options=all_specs,
                default=top_specs,
            )

        # ---- Right: Numeric feature selection ----
        with col2:
            # Salary'yi isteğe bağlı bırakmak için varsayılan exclude
            exclude = {"ID", "Salary", "SalaryCategory"}
            available = [
                c
                for c in df.select_dtypes(include=["number"]).columns
                if c not in exclude
            ]

            # Varsayılan olarak göstereceğimiz feature seti:
            # Academic + Aptitude + Technical + Personality
            defaults = [
                # Academic history
                "collegeGPA",
                "10percentage",
                "12percentage",
                # Aptitude tests
                "English",
                "Logical",
                "Quant",
                # Technical skills
                "Domain",
                "ComputerProgramming",
                # Personality traits (Big Five style, CSV'deki isimleriyle)
                "conscientiousness",
                "agreeableness",
                "extraversion",
                "nueroticism",            # dataset'teki yazımı
                "openess_to_experience",  # dataset'teki yazımı
            ]
            defaults = [c for c in defaults if c in available]

            selected_features = st.multiselect(
                "Features to Analyze",
                options=available,
                default=defaults,
                help=(
                    "You can also add Salary or other numeric columns here "
                    "if you want to see their correlations."
                ),
            )
            show_values = st.checkbox("Show correlation values", value=True)

    # ─────────────────────── Apply filters ───────────────────────
    df_filtered = df[
        df["Degree"].isin(selected_degree)
        & df["Specialization"].isin(selected_specs)
    ]

    if df_filtered.empty or len(selected_features) < 2:
        st.warning("No data or not enough features selected.")
        return

    # ───────────────── Correlation & Clustering ─────────────────
    corr_matrix = df_filtered[selected_features].corr().round(2)

    try:
        ordered_cols = compute_clustered_order(corr_matrix)
        corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
    except Exception as e:
        st.warning(f"Clustering failed, showing unclustered matrix. Error: {e}")

    # ─────────────────────── Visualization ───────────────────────
    st.divider()
    st.subheader("🔥 Clustered Correlation Heatmap")

    fig = px.imshow(
        corr_matrix,
        text_auto=show_values,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
    )

    fig.update_layout(
        height=700,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis=dict(side="bottom", tickangle=-45),
        modebar=dict(
            bgcolor="rgba(0,0,0,0.3)",
            color="white",
            activecolor="#00D9FF",
        ),
    )

    if show_values:
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=12)

    # Plotly config for better interactivity / export
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "gpa_skills_personality_clustering_heatmap",
            "height": 1080,
            "width": 1920,
            "scale": 2,
        },
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

    # ───────────────────────── Insights ─────────────────────────
    st.markdown("### 💡 Key Insights")

    pairs = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    ).stack().reset_index()
    pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    pairs["AbsCorr"] = pairs["Correlation"].abs()
    top_pairs = pairs.sort_values("AbsCorr", ascending=False).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strongest Correlations (by absolute value):**")
        for _, row in top_pairs.iterrows():
            color = "green" if row["Correlation"] > 0 else "red"
            st.markdown(
                f"- **{row['Feature 1']}** & **{row['Feature 2']}**: "
                f":{color}[**{row['Correlation']:.2f}**]"
            )

    with col2:
        st.caption(
            """
            **Color scale:**  
            - Warm colors (red) ≈ strong positive correlation  
            - Cool colors (blue) ≈ strong negative correlation  

            Academic scores, aptitude tests, technical skills and
            personality traits tend to form their own clusters,
            revealing natural student/talent profiles.
            """
        )


if __name__ == "__main__":
    main()
