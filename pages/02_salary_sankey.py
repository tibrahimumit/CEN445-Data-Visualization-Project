from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from data_loader import load_data


def ensure_helper_columns(df):
    """Guarantee tier and salary label columns exist even if cache persisted older data."""
    if "CollegeTierLabel" not in df.columns:
        df["CollegeTierLabel"] = df["CollegeTier"].apply(
            lambda tier: f"Tier {int(tier)}" if tier == tier else "Tier Unknown"
        )

    if "SalaryCategory" not in df.columns:
        def categorize(value):
            if value != value:
                return "Salary Unknown"
            if value < 50_000:
                return "Low (<50k)"
            if value <= 100_000:
                return "Medium (50k-100k)"
            return "High (>100k)"

        df["SalaryCategory"] = df["Salary"].apply(categorize)

    return df


def build_sankey(df) -> go.Figure:
    """Create Sankey inputs across four categorical stages."""
    degree_nodes = sorted(df["Degree"].dropna().unique())
    spec_nodes = sorted(df["Specialization"].dropna().unique())
    tier_nodes = sorted(df["CollegeTierLabel"].dropna().unique())
    salary_nodes = ["Low (<50k)", "Medium (50k-100k)", "High (>100k)"]

    labels = degree_nodes + spec_nodes + tier_nodes + salary_nodes
    node_index = {label: idx for idx, label in enumerate(labels)}

    def pair_counts(left_col: str, right_col: str):
        subset = df[[left_col, right_col]].dropna()
        return subset.groupby([left_col, right_col]).size().reset_index(name="value")

    source, target, values, hovertexts = [], [], [], []

    for left, right in [
        ("Degree", "Specialization"),
        ("Specialization", "CollegeTierLabel"),
        ("CollegeTierLabel", "SalaryCategory"),
    ]:
        for _, row in pair_counts(left, right).iterrows():
            l_val, r_val, cnt = row[left], row[right], row["value"]
            if l_val not in node_index or r_val not in node_index:
                continue
            source.append(node_index[l_val])
            target.append(node_index[r_val])
            values.append(int(cnt))
            hovertexts.append(f"{l_val} → {r_val}: {cnt} students")

    color_palette = ["#0091AD", "#4CC9F0", "#F72585", "#FFD166", "#06D6A0", "#118AB2"]
    node_colors = [color_palette[i % len(color_palette)] for i in range(len(labels))]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="#1f1f1f", width=1),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                    hovertemplate="%{customdata}",
                    customdata=hovertexts,
                    color="rgba(136, 136, 136, 0.45)",
                ),
            )
        ]
    )

    fig.update_layout(
        height=650,
        margin=dict(l=30, r=30, t=40, b=40),
        font=dict(color="#FFFFFF"),
        plot_bgcolor="#050816",
        paper_bgcolor="#050816",
        modebar=dict(
            bgcolor="rgba(0,0,0,0.3)",
            color="white",
            activecolor="#00D9FF"
        ),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Sankey: Academic Flow to Salary", layout="wide")
    st.title("Sankey Diagram: Degree → Specialization → College Tier → Salary")

    st.markdown(
        """
        Track how academic decisions cascade into salary outcomes. Each band shows the number of students
        flowing through **Degree → Specialization → College Tier → Salary Category**.
        Use the filters below to isolate specific cohorts and inspect the corresponding flows.
        """
    )

    df = ensure_helper_columns(load_data())

    cols = st.columns(4)
    with cols[0]:
        selected_degree = st.multiselect(
            "Degree",
            options=sorted(df["Degree"].dropna().unique()),
            default=sorted(df["Degree"].dropna().unique()),
        )
    with cols[1]:
        selected_spec = st.multiselect(
            "Specialization",
            options=sorted(df["Specialization"].dropna().unique()),
            default=sorted(df["Specialization"].dropna().unique()),
        )
    with cols[2]:
        selected_tier = st.multiselect(
            "College tier",
            options=sorted(df["CollegeTierLabel"].dropna().unique()),
            default=sorted(df["CollegeTierLabel"].dropna().unique()),
        )
    with cols[3]:
        salary_min = float(df["Salary"].min())
        salary_max = float(df["Salary"].max())
        selected_range = st.slider(
            "Salary filter (₹ per year)",
            min_value=salary_min,
            max_value=salary_max,
            value=(salary_min, salary_max),
            step=10_000.0,
        )

    filtered = df[
        df["Degree"].isin(selected_degree)
        & df["Specialization"].isin(selected_spec)
        & df["CollegeTierLabel"].isin(selected_tier)
        & df["Salary"].between(selected_range[0], selected_range[1])
    ]

    if filtered.empty:
        st.warning("No students match the current filters. Try broadening the selections.")
        return

    fig = build_sankey(filtered)
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'salary_sankey',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)

    st.caption(
        "Tip: Hover over any band to see exact counts. The final nodes (Low / Medium / High) "
        "are derived from salary buckets: Low < ₹50k, Medium ₹50–100k, High > ₹100k."
    )

if __name__ == "__main__":
    main()


