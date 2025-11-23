import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_data

DEFAULT_USD_RATE = 83.0


def plot_parallel_coordinates(df: pd.DataFrame) -> None:
    """Render the parallel coordinates chart with the specified columns."""
    st.subheader("Parallel Coordinates: Academic & Aptitude Scores vs Salary")

    st.markdown(
        """
        This chart compares each student's high-school scores (10th & 12th grade), 
        university GPA, and aptitude test scores (English & Quantitative) while encoding 
        annual salary as the color dimension. Every polyline equals one graduate; its path 
        across the axes shows the profile and the color reveals salary intensity.
        """
    )

    dims = [
        "10percentage",
        "12percentage",
        "collegeGPA",
        "English",
        "Quant",
    ]

    required_cols = dims + ["Salary", "PlacementStatus", "Degree", "Specialization"]
    df_pc = df.dropna(subset=required_cols)

    with st.expander("Filters", expanded=True):
        placement_opts = sorted(df_pc["PlacementStatus"].dropna().unique())
        degree_opts = sorted(df_pc["Degree"].dropna().unique())
        spec_opts = sorted(df_pc["Specialization"].dropna().unique())

        st.markdown("#### Profile selectors")
        col_p, col_d = st.columns(2)
        with col_p:
            selected_placement = st.multiselect(
                "Placement status",
                options=placement_opts,
                default=placement_opts,
                placeholder="Choose Placed / Not Placed…",
                help="Pick any combination to focus on specific placement outcomes.",
            )
        with col_d:
            selected_degree = st.multiselect(
                "Undergraduate degree",
                options=degree_opts,
                default=degree_opts,
                placeholder="Choose B.Tech, M.Sc, etc…",
            )

        st.markdown("#### Specialization / major")
        selected_spec = st.multiselect(
            "Major selection",
            options=spec_opts,
            default=spec_opts,
            placeholder="Choose computer eng., electronics, etc…",
        )

        min_salary = float(df_pc["Salary"].min())
        max_salary = float(df_pc["Salary"].max())
        salary_range = st.slider(
            "Salary range (₹ / year)",
            min_value=min_salary,
            max_value=max_salary,
            value=(min_salary, max_salary),
            step=1_000.0,
        )
        st.caption(
            f"Range in INR: ₹{salary_range[0]:,.0f} – ₹{salary_range[1]:,.0f} per year"
        )

    with st.popover("🎨 Color scale legend"):
        st.markdown("Higher salaries shift from deep indigo to neon yellow.")
        palette_css = """
        <div style="
            width: 100%;
            height: 20px;
            border-radius: 6px;
            background: linear-gradient(90deg,
                #030637,
                #053A5F,
                #055F8C,
                #00A6CE,
                #00E5FF,
                #6FFFE9,
                #F9FF8B
            );
            border: 1px solid #333;
        "></div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#ccc; margin-top:6px;">
            <span>Lower salary</span>
            <span>Higher salary</span>
        </div>
        """
        st.markdown(palette_css, unsafe_allow_html=True)

    with st.popover("💱 Salary FX conversion"):
        usd_rate = st.number_input(
            "USD / INR rate",
            min_value=50.0,
            max_value=120.0,
            value=DEFAULT_USD_RATE,
            step=0.5,
            help="Update this value to refresh every salary conversion.",
        )
        st.markdown(
            f"""
            - Current INR range: **₹{salary_range[0]:,.0f} – ₹{salary_range[1]:,.0f}**
            - Converted range: **${salary_range[0]/usd_rate:,.0f} – ${salary_range[1]/usd_rate:,.0f}**
            """
        )

    mask = (
        df_pc["PlacementStatus"].isin(selected_placement)
        & df_pc["Degree"].isin(selected_degree)
        & df_pc["Specialization"].isin(selected_spec)
        & df_pc["Salary"].between(*salary_range)
    )

    df_filtered = df_pc[mask]

    if df_filtered.empty:
        st.warning("No records match the current filters. Try widening them.")
        return

    max_lines = st.slider(
        "Maximum polylines to draw:",
        min_value=50,
        max_value=len(df_filtered),
        value=min(600, len(df_filtered)),
        step=1,
        help="Sampling keeps the chart legible when many students are selected.",
    )

    if len(df_filtered) > max_lines:
        df_plot = df_filtered.sample(max_lines, random_state=42)
    else:
        df_plot = df_filtered

    labels_tr = {
        "10percentage": "Grade 10 %",
        "12percentage": "Grade 12 %",
        "collegeGPA": "University GPA",
        "English": "English score",
        "Quant": "Quant score",
        "Salary": "Salary",
    }

    electric_scale = [
        "#030637",
        "#053A5F",
        "#055F8C",
        "#00A6CE",
        "#00E5FF",
        "#6FFFE9",
        "#F9FF8B",
    ]

    fig = px.parallel_coordinates(
        df_plot,
        dimensions=dims,
        color="Salary",
        color_continuous_scale=electric_scale,
        labels=labels_tr,
        range_color=[df_plot["Salary"].min(), df_plot["Salary"].max()],
    )

    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#050816",
        paper_bgcolor="#02030A",
        font=dict(color="white"),
        modebar=dict(
            bgcolor="rgba(0,0,0,0.3)",
            color="white",
            activecolor="#00D9FF"
        ),
    )

    fig.update_traces(
        line=dict(coloraxis=None),
        selector=dict(type="parcoords"),
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text="Salary (INR)", side="right"),
            tickprefix="₹",
            len=0.75,
            thickness=18,
            bgcolor="#111111",
            bordercolor="#444444",
            borderwidth=1,
        )
    )

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'parallel_coordinates',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)

    st.caption(
        "Each polyline corresponds to a graduate. Drag along the axes to brush specific profiles "
        "and use the neon palette to pinpoint which attribute mixes correlate with higher pay."
    )

    usd_min = df_filtered["Salary"].min() / usd_rate
    usd_max = df_filtered["Salary"].max() / usd_rate
    usd_avg = df_filtered["Salary"].mean() / usd_rate

    col1, col2, col3 = st.columns(3)
    col1.metric("Min salary (USD / year)", f"${usd_min:,.0f}")
    col2.metric("Average salary (USD / year)", f"${usd_avg:,.0f}")
    col3.metric("Max salary (USD / year)", f"${usd_max:,.0f}")


def main() -> None:
    df = load_data()
    plot_parallel_coordinates(df)


if __name__ == "__main__":
    main()


