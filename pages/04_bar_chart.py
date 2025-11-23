"""
Specialization Distribution - Bar Chart

Shows student distribution across specializations with icons and dynamic colors.
Also shows stacked range distribution (GPA / English / Quant / etc.) within top specializations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_data


SPECIALIZATION_ICONS = {
    "Computer Science": "🖥️", "Computer Engineering": "💻", "Information Technology": "📱",
    "Software Engineering": "⌨️", "Mechanical Engineering": "⚙️", "Electrical Engineering": "🔌",
    "Electronics Engineering": "🔋", "Civil Engineering": "🏗️", "Chemical Engineering": "🧪",
    "Biotechnology": "🧬", "Electronics and Communication": "📡", "Instrumentation Engineering": "🎛️",
}


def get_icon(spec: str) -> str:
    """Get icon for specialization."""
    return SPECIALIZATION_ICONS.get(spec, "📚")


def main():
    st.set_page_config(
        page_title="Specialization Distribution",
        layout="wide",
        page_icon="📊"
    )
    
    df = load_data()
    
    st.title("📊 Student Distribution by Specialization")
    st.markdown(
        "Interactive bar chart showing the number of students in each specialization. "
        "Use the filters to switch between a simple count view and a detailed score range distribution."
    )

    # ---- Score metrics that can be used for range distribution ----
    score_label_map = {
        "collegeGPA": "College GPA",
        "English": "English Score",
        "Logical": "Logical Reasoning",
        "Quant": "Quantitative Score",
        "ComputerProgramming": "Programming Score",
        "Programming": "Programming Score",
    }
    # Sadece gerçekten dataset'te olan kolonları al
    available_score_cols = [c for c in score_label_map.keys() if c in df.columns]
    if not available_score_cols:
        available_score_cols = ["collegeGPA"] if "collegeGPA" in df.columns else []

    # Filters
    with st.expander("🔧 Filters & Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            degree_opts = sorted(df["Degree"].dropna().unique())
            selected_degree = st.multiselect("Degree", degree_opts, default=degree_opts)
            
            tier_opts = sorted(df["CollegeTier"].dropna().unique())
            selected_tier = st.multiselect("College Tier", tier_opts, default=selected_tier if 'selected_tier' in locals() else tier_opts)

        with col2:
            top_n = st.slider("Number of Specializations (Top N)", min_value=3, max_value=20, value=10)
            orientation = st.radio("Orientation", ["Horizontal", "Vertical"], index=0)

            chart_mode = st.radio(
                "Chart Mode",
                ["Simple Count", "Score Range Distribution"],
                index=0
            )

            if chart_mode == "Score Range Distribution":
                score_metric = st.selectbox(
                    "Select Score Metric",
                    options=available_score_cols,
                    format_func=lambda c: score_label_map.get(c, c),
                )
            else:
                score_metric = None

            if "Gender" in df.columns:
                gender_opts = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
                selected_gender = st.selectbox("Gender Filter", gender_opts, index=0)
            else:
                selected_gender = "All"

    # Apply filters
    mask = df["Degree"].isin(selected_degree) & df["CollegeTier"].isin(selected_tier)
    if selected_gender != "All":
        mask &= (df["Gender"] == selected_gender)
    
    df_filtered = df[mask]
    
    if df_filtered.empty:
        st.warning("No data available. Adjust filters.")
        return

    # Aggregate by specialization (base bar chart)
    spec_counts = (
        df_filtered.groupby("Specialization")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("Students")
        .reset_index()
    )
    
    # Add icons
    spec_counts["Icon"] = spec_counts["Specialization"].apply(get_icon)
    spec_counts["Display"] = spec_counts["Icon"] + " " + spec_counts["Specialization"]

    # === VISUALIZATION LOGIC ===
    st.divider()
    
    if chart_mode == "Simple Count":
        st.subheader(f"📊 Top {top_n} Specializations by Student Count")
        
        if orientation == "Horizontal":
            fig = px.bar(
                spec_counts,
                x="Students",
                y="Display",
                orientation="h",
                color="Students",
                color_continuous_scale="Blues",
                text="Students",
            )
            fig.update_layout(yaxis_title="", xaxis_title="Number of Students")
        else:
            fig = px.bar(
                spec_counts,
                x="Display",
                y="Students",
                color="Students",
                color_continuous_scale="Blues",
                text="Students",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Number of Students", xaxis_tickangle=-45)
        
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=max(500, len(spec_counts) * 40),
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            showlegend=False,
            modebar=dict(bgcolor="rgba(0,0,0,0.3)", color="white", activecolor="#00D9FF"),
        )
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        # === SCORE RANGE DISTRIBUTION MODE ===
        metric_label = score_label_map.get(score_metric, score_metric)
        st.subheader(f"🎓 {metric_label} Range Distribution within Top Specializations")

        # Filter for top specializations
        top_specs = spec_counts["Specialization"].tolist()
        df_metric = df_filtered[df_filtered["Specialization"].isin(top_specs)].copy()

        # Clean metric column
        df_metric[score_metric] = df_metric[score_metric].replace(-1, np.nan)
        df_metric = df_metric.dropna(subset=[score_metric])

        if df_metric.empty:
            st.info(f"No data available for {metric_label} after cleaning.")
            return

        # Binning logic
        if score_metric == "collegeGPA":
            bins = [0, 60, 70, 80, 90, 100.0001]
            labels = ["<60", "60–70", "70–80", "80–90", "90–100"]
        else:
            metric_min = df_metric[score_metric].min()
            metric_max = df_metric[score_metric].max()
            
            if metric_min == metric_max:
                st.info(f"{metric_label} has no variation for selected filters.")
                return

            bins = np.linspace(metric_min, metric_max, 6)
            labels = [f"{int(bins[i])}–{int(bins[i+1])}" for i in range(len(bins) - 1)]

        range_col = f"{metric_label} Range"
        df_metric[range_col] = pd.cut(
            df_metric[score_metric],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        # Aggregate
        metric_counts = (
            df_metric.groupby(["Specialization", range_col])
            .size()
            .reset_index()
            .rename(columns={0: "Students"})
        )

        # Merge display names
        metric_counts = metric_counts.merge(
            spec_counts[["Specialization", "Icon", "Display"]],
            on="Specialization",
            how="left",
        )

        range_order = list(labels)

        if orientation == "Horizontal":
            fig_metric = px.bar(
                metric_counts,
                x="Students",
                y="Display",
                color=range_col,
                orientation="h",
                category_orders={range_col: range_order},
                text="Students",
            )
            fig_metric.update_layout(yaxis_title="", xaxis_title="Number of Students")
        else:
            fig_metric = px.bar(
                metric_counts,
                x="Display",
                y="Students",
                color=range_col,
                category_orders={range_col: range_order},
                text="Students",
            )
            fig_metric.update_layout(xaxis_title="", yaxis_title="Number of Students", xaxis_tickangle=-45)

        fig_metric.update_traces(textposition="inside", textfont_size=11, insidetextanchor="middle")
        fig_metric.update_layout(
            barmode="stack",
            height=max(500, len(spec_counts) * 45),
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            legend_title_text=f"{metric_label} Range",
            modebar=dict(bgcolor="rgba(0,0,0,0.3)", color="white", activecolor="#00D9FF"),
        )

        st.plotly_chart(fig_metric, use_container_width=True)

    # === KEY FINDINGS ===
    st.divider()
    st.subheader("💡 Key Findings")

    if chart_mode == "Simple Count":
        top_spec_name = spec_counts.iloc[0]["Specialization"]
        top_spec_count = spec_counts.iloc[0]["Students"]
        total_students = spec_counts["Students"].sum()
        share = (top_spec_count / total_students) * 100

        st.markdown(
            f"""
            - **Most Popular:** The top specialization is **{top_spec_name}** with **{top_spec_count:,}** students.
            - **Dominance:** It accounts for **{share:.1f}%** of the students in the top {top_n} categories.
            - **Total:** A total of **{total_students:,}** students are represented in this view.
            """
        )
    
    else:
        # Score Range Findings
        # Find the range with max students for the top specialization
        top_spec_row = metric_counts[metric_counts["Specialization"] == spec_counts.iloc[0]["Specialization"]]
        if not top_spec_row.empty:
            best_range_row = top_spec_row.loc[top_spec_row["Students"].idxmax()]
            range_name = best_range_row[range_col]
            count_in_range = best_range_row["Students"]
            
            st.markdown(
                f"""
                - **Top Specialization Insight:** In **{spec_counts.iloc[0]['Specialization']}**, 
                  the most common **{metric_label}** range is **{range_name}** 
                  (with **{count_in_range}** students).
                - **Distribution:** This chart helps identify if certain specializations tend to have 
                  higher or lower scores in **{metric_label}**.
                """
            )
        else:
            st.markdown("No sufficient data to generate insights for the top specialization.")


if __name__ == "__main__":
    main()
